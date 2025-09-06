import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

final class CategoricalDataClassifierTest {
  private CategoricalDataClassifierTest() {
  }

  public static void main(String[] args) throws Exception {
    List<Pair<StringRow, String>> dataTrain, dataTest;
    String[] columnNames;
    int[] inFeatures;
    int outFeature;
    final int trainSize, testSize;

    try (final Scanner sc = new Scanner(System.in)) {
      {
        System.out.print("Enter dataset CSV file: ");
        final var original = StringDataset.readCSV(sc.nextLine());
        System.out.println("Reading...");
        columnNames = original.getColumnNames();
        System.out.println("Number of rows: " + original.size());
        System.out.println("Columns: " + Arrays.toString(columnNames));

        System.out.print("Use randomized seed?[y/n]: ");
        final var useRandomizedSeed = Character.toLowerCase(sc.nextLine().trim().charAt(0)) == 'y';
        // My birthday as my seed
        final Random rng;
        if (useRandomizedSeed)
          rng = new Random();
        else {
          System.out.print("Enter 64-bit signed integer for seed: ");
          rng = new Random(Long.parseLong(sc.nextLine().trim()));
        }
        StringDataset originalTrain, originalTest;
        System.out.print("Enter split fraction (between 0 to 1): ");
        final double splitFraction = Double.parseDouble(sc.nextLine().trim());
        {
          final var split = Dataset.split(original, rng, splitFraction).parallel()
              .map(p -> new StringDataset(columnNames, p.parallel()))
              .collect(Collectors.toUnmodifiableList());
          assert split.size() == 2;
          originalTrain = split.get(0);
          originalTest = split.get(1);
        }
        trainSize = originalTrain.size();
        testSize = originalTest.size();
        System.out.printf("Training size: %d, Test size: %d\n", trainSize, testSize);
        assert trainSize + testSize == original.size();

        inFeatures = IntStream.range(0, original.getNumColumns() - 1).toArray();
        outFeature = original.getNumColumns() - 1;

        dataTrain = originalTrain.parallelStream()
            .map(row -> new Pair<>(row.project(inFeatures), row.get(outFeature)))
            .collect(Collectors.toList());
        dataTest = originalTest.parallelStream()
            .map(row -> new Pair<>(row.project(inFeatures), row.get(outFeature)))
            .collect(Collectors.toList());
      }

      System.gc();
      System.out.println();

      {
        System.out.print("Enter decision tree depth limit: ");
        final int depthLimit = Integer.parseInt(sc.nextLine().trim());

        final var attrKinds = Arrays.stream(inFeatures).mapToObj(i -> AttrKind.CATEGORICAL).toArray(AttrKind[]::new);
        final var dtree = new DecisionTreeClassifier<StringRow, String>(attrKinds, columnNames, depthLimit,
            m -> Utils.countedEntropy(m.values().stream().unordered()));

        testClassifier(dtree, dataTrain, dataTest);

        System.out.println("\nTree structure:");
        dtree.walkTree(System.out);
      }
    }
  }

  static <IR extends Row, OP> void testClassifier(
      Classifier<IR, OP> classifier,
      List<Pair<IR, OP>> dataTrain,
      List<Pair<IR, OP>> dataTest) {
    final int trainSize = dataTrain.size();
    final int testSize = dataTest.size();

    System.out.println(classifier.getClass().getName() + " fitting starting...");
    long ts = System.currentTimeMillis();
    classifier.fit(dataTrain.iterator());
    classifier.finishFitting();
    ts = System.currentTimeMillis() - ts;
    System.out.println("Fitting took " + ts + "ms time");

    System.out.println(classifier.getClass().getName() + " evaluation starting...");

    ts = System.currentTimeMillis();
    int trainScore = (int) classifier.countCorrectParallel(dataTrain.parallelStream().unordered(), null);
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Training accuracy score: %f%% (%d out of %d); took %dms time\n", trainScore * 100D / trainSize,
        trainScore, trainSize, ts);

    ts = System.currentTimeMillis();
    int testScore = (int) classifier.countCorrectParallel(dataTest.parallelStream().unordered(), null);
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Test accuracy score: %f%% (%d out of %d); took %dms time\n", testScore * 100D / testSize,
        testScore,
        testSize, ts);
  }
}
