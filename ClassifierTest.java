import java.util.*;
import java.util.stream.*;

final class ClassifierTest {
  private ClassifierTest() {
  }

  public static void main(String args[]) throws Exception {
    final List<Integer> outcomeTrain, outcomeTest;
    final List<Float64Row> inputTrain, inputTest;
    final String[] columnNames;
    final int[] inFeatures;
    final int outFeature;
    final int trainSize, testSize;

    System.out.println("Reading...");
    {
      final var original = StringDataset.readCSV(args.length > 0 ? args[0] : "diabetes.csv");
      columnNames = original.getColumnNames();
      System.out.println("Number of rows: " + original.size());
      System.out.println("Columns: " + Arrays.toString(columnNames));

      final Random rng = new Random(0x12012001_12012001L); // my birthday as my seed
      StringDataset originalTrain, originalTest;
      final double splitFraction = args.length > 1 ? Double.parseDouble(args[1]) : 0.75;
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

      // Input feature indices
      inFeatures = IntStream.range(0, original.getNumColumns() - 1).toArray();
      outFeature = original.getNumColumns() - 1;
      outcomeTrain = originalTrain.columnStream(outFeature).parallel()
          .map(String::trim).map(Integer::valueOf).collect(Collectors.toUnmodifiableList());
      outcomeTest = originalTest.columnStream(outFeature).parallel()
          .map(String::trim).map(Integer::valueOf).collect(Collectors.toUnmodifiableList());
      inputTrain = originalTrain.stream().parallel()
          .map(row -> new Float64Row(row.project(inFeatures)))
          .collect(Collectors.toUnmodifiableList());
      inputTest = originalTest.stream().parallel()
          .map(row -> new Float64Row(row.project(inFeatures)))
          .collect(Collectors.toUnmodifiableList());
      // original = null; originalTrain = null; originalTest = null;
    }

    System.gc();
    System.out.println();

    testClassifier(new MinimumDistanceClassifier<>(), inputTrain, outcomeTrain, inputTest, outcomeTest);

    System.gc();
    System.out.println();

    testClassifier(new KNearestNeighbors<>(args.length > 2 ? Integer.parseInt(args[2]) : 5), inputTrain, outcomeTrain, inputTest, outcomeTest);
  }

  static <IR extends Row, OP> void testClassifier(
      Classifier<IR, OP> classifier,
      List<? extends IR> inputTrain,
      List<? extends OP> outcomeTrain,
      List<? extends IR> inputTest,
      List<? extends OP> outcomeTest) {
    final int trainSize = inputTrain.size();
    final int testSize = inputTest.size();

    System.out.println(classifier.getClass().getName()+" fitting starting...");
    long ts = System.currentTimeMillis();
    classifier.fit(inputTrain.iterator(), outcomeTrain.iterator());
    ts = System.currentTimeMillis() - ts;
    System.out.println("Fitting took " + ts + "ms time");

    System.out.println(classifier.getClass().getName()+" evaluation starting...");

    ts = System.currentTimeMillis();
    int trainScore = (int) classifier.countCorrectParallel(
        IntStream.range(0, trainSize).unordered().parallel()
            .mapToObj(i -> new Classifier.Pair<IR, OP>(inputTrain.get(i), outcomeTrain.get(i))),
        null /* cnt -> System.out.printf("%d entries checked out of %d\r", cnt, trainSize) */);
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Training score: %f%% (%d out of %d); took %dms time\n", trainScore * 100D / trainSize,
        trainScore, trainSize, ts);

    ts = System.currentTimeMillis();
    int testScore = (int) classifier.countCorrectParallel(
        IntStream.range(0, testSize).unordered().parallel()
            .mapToObj(i -> new Classifier.Pair<IR, OP>(inputTest.get(i), outcomeTest.get(i))),
        null /* cnt -> System.out.printf("%d entries checked out of %d\r", cnt, testSize) */);
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Test score: %f%% (%d out of %d); took %dms time\n", testScore * 100D / testSize, testScore,
        testSize, ts);
  }
}
