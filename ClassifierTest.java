import java.util.*;
import java.util.stream.*;

final class ClassifierTest {
  private ClassifierTest() {
  }

  public static void main(String args[]) throws Exception {
    final List<String> outcomeTrain, outcomeTest;
    final List<Float64Row> inputTrain, inputTest;
    final String[] columnNames;
    final int[] inFeatures;
    final int outFeature;
    final int trainSize, testSize;
    final Random rng;

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
        // final Random rng;
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

        // Input feature indices
        inFeatures = IntStream.range(0, original.getNumColumns() - 1).toArray();
        outFeature = original.getNumColumns() - 1;
        outcomeTrain = originalTrain.columnStream(outFeature).parallel()
            .map(String::trim).collect(Collectors.toUnmodifiableList());
        outcomeTest = originalTest.columnStream(outFeature).parallel()
            .map(String::trim).collect(Collectors.toUnmodifiableList());
        inputTrain = originalTrain.stream().parallel()
            .map(row -> new Float64Row(row.project(inFeatures)))
            .collect(Collectors.toCollection(ArrayList::new));
        inputTest = originalTest.stream().parallel()
            .map(row -> new Float64Row(row.project(inFeatures)))
            .collect(Collectors.toCollection(ArrayList::new));

        System.out.print("Use IQR scaling?[y/n]: ");
        final boolean useRobust = Character.toLowerCase(sc.nextLine().trim().charAt(0)) == 'y';
        if (useRobust) {
          final var robustScaler = new RobustScaler(inFeatures.length);
          robustScaler.fit(inputTrain.iterator());
          robustScaler.fit(inputTest.iterator());
          robustScaler.finishFitting();
          Float64Dataset.inplaceTransform(inputTrain, robustScaler);
          Float64Dataset.inplaceTransform(inputTest, robustScaler);
        }

        System.out.print("Use standardization?[y/n]: ");
        final boolean useStandardization = Character.toLowerCase(sc.nextLine().trim().charAt(0)) == 'y';
        if (useStandardization) {
          final var stdScaler = new StandardScaler(inFeatures.length);
          stdScaler.fit(inputTrain.iterator());
          stdScaler.fit(inputTest.iterator());
          stdScaler.finishFitting();
          Float64Dataset.inplaceTransform(inputTrain, stdScaler);
          Float64Dataset.inplaceTransform(inputTest, stdScaler);
        }

        // original = null; originalTrain = null; originalTest = null;
      }

      System.gc();
      System.out.println();

      testClassifier(new RandomClassifier<>(rng), inputTrain, outcomeTrain, inputTest, outcomeTest);

      System.gc();
      System.out.println();

      testClassifier(new MinimumDistanceClassifier<>(Float64Row::distanceChebyshev), inputTrain, outcomeTrain,
          inputTest, outcomeTest);

      System.gc();
      System.out.println();

      System.out.print("Enter decision tree real value binning splits: ");
      final int realAttributeSplits = Integer.parseInt(sc.nextLine().trim());
      System.out.print("Enter decision tree max depth: ");
      final int depthLimit = Integer.parseInt(sc.nextLine().trim());
      System.out.print("Enter minimum number of samples to split nodes: ");
      final int minSamplesToSplit = Integer.parseInt(sc.nextLine().trim());
      final AttrKind[] attrKinds = IntStream.range(0, inFeatures.length).mapToObj(i -> AttrKind.CONTINUOUS)
          .toArray(AttrKind[]::new);
      {
        final var dtree = new DecisionTreeClassifier<Float64Row, String>(attrKinds, columnNames, depthLimit,
            realAttributeSplits, minSamplesToSplit, m -> Utils.countedEntropy(m.values().stream().unordered()));
        try {
          testClassifier(dtree, inputTrain, outcomeTrain, inputTest, outcomeTest);
        } catch (RuntimeException ex) {
          ex.printStackTrace(System.err);
        }
        // System.out.println("\nTree structure:");
        // dtree.walkTree(System.out);
      }

      System.gc();
      System.out.println();

      System.out.print("Enter learning rate for perceptrons: ");
      final double learningRate = Double.parseDouble(sc.nextLine().trim());
      System.out.print("Enter maximum number of epochs: ");
      final long maxEpochs = Long.parseLong(sc.nextLine().trim());
      {
        final var perceptron = new PerceptronClassifier<String>(inFeatures.length, learningRate, maxEpochs, -1d,
            ActivationFunction.IDENTITY, LossFunction.SOFTMAX_LOG_LOSS, rng);
        perceptron.setEpochLossNotifier((e, l) -> System.out.printf("epoch %d/%d; loss %f\r", e, maxEpochs, l));
        testClassifier(perceptron, inputTrain, outcomeTrain, inputTest, outcomeTest);
      }

      System.gc();
      System.out.println();

      System.out.print("Enter hidden layer sizes separated by space: ");
      final var hiddenLayerSizes = Stream.of(sc.nextLine().split("\\s"))
          .filter(s -> !s.isBlank())
          .mapToInt(Integer::parseInt).toArray();
      System.out.println("Number of hidden layers: " + hiddenLayerSizes.length);
      {
        final var mlp = new MLPClassifier<String>(
            inFeatures.length,
            hiddenLayerSizes,
            learningRate,
            maxEpochs,
            Double.MIN_NORMAL,
            ActivationFunction.SIGMOID,
            ActivationFunction.SIGMOID,
            LossFunction.SOFTMAX_LOG_LOSS,
            rng);
        mlp.setEpochLossNotifier((e, l) -> System.out.printf("epoch %d/%d; loss %f\r", e, maxEpochs, l));
        testClassifier(mlp, inputTrain, outcomeTrain, inputTest, outcomeTest);
      }

      System.gc();
      System.out.println();

      System.out.print("Enter KNN parameter K: ");
      final int K = Integer.parseInt(sc.nextLine().trim());
      System.out.println("KNN parameter K: " + K);
      testClassifier(new KNearestNeighbors<>(K, Float64Row::distanceChebyshev), inputTrain, outcomeTrain, inputTest,
          outcomeTest);
    }
  }

  static <IR extends Row, OP> void testClassifier(
      Classifier<IR, OP> classifier,
      List<? extends IR> inputTrain,
      List<? extends OP> outcomeTrain,
      List<? extends IR> inputTest,
      List<? extends OP> outcomeTest) {
    final int trainSize = inputTrain.size();
    final int testSize = inputTest.size();

    System.out.println(classifier.getClass().getName() + " fitting starting...");
    long ts = System.currentTimeMillis();
    classifier.fit(inputTrain.iterator(), outcomeTrain.iterator());
    classifier.finishFitting();
    ts = System.currentTimeMillis() - ts;
    System.out.println("Fitting took " + ts + "ms time");

    if (classifier instanceof DecisionTreeClassifier<?, ?> dtree) {
      System.out.println("Tree structure:");
      try {
        dtree.walkTree(System.out);
      } catch (java.io.IOException ex) {
      }
      System.out.println();
    }

    System.out.println(classifier.getClass().getName() + " evaluation starting...");

    ts = System.currentTimeMillis();
    int trainScore = (int) classifier.countCorrectParallel(
        IntStream.range(0, trainSize).unordered().parallel()
            .mapToObj(i -> new Pair<IR, OP>(inputTrain.get(i), outcomeTrain.get(i))),
        cnt -> System.out.printf("%d entries checked out of %d\r", cnt, trainSize));
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Training accuracy score: %f%% (%d out of %d); took %dms time\n", trainScore * 100D / trainSize,
        trainScore, trainSize, ts);

    ts = System.currentTimeMillis();
    int testScore = (int) classifier.countCorrectParallel(
        IntStream.range(0, testSize).unordered().parallel()
            .mapToObj(i -> new Pair<IR, OP>(inputTest.get(i), outcomeTest.get(i))),
        cnt -> System.out.printf("%d entries checked out of %d\r", cnt, testSize));
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Test accuracy score: %f%% (%d out of %d); took %dms time\n", testScore * 100D / testSize,
        testScore,
        testSize, ts);
  }
}
