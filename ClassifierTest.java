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

    System.gc(); System.out.println();

    {
      var classifier = new MinimumDistanceClassifier<Integer>();

      System.out.println("mindist fitting starting...");
      long ts = System.currentTimeMillis();
      classifier.fit(inputTrain, outcomeTrain);
      ts = System.currentTimeMillis() - ts;
      System.out.println("Fitting took " + ts + "ms time");

      System.out.println("mindist evaluation starting...");
      final var counter = new java.util.concurrent.atomic.AtomicInteger(0);
      ts = System.currentTimeMillis();
      int trainScore = (int) IntStream.range(0, trainSize).unordered().parallel()
          .peek(i -> System.out.printf("%d entries checked out of %d\r", counter.incrementAndGet(), trainSize))
          .filter(i -> Objects.equals(outcomeTrain.get(i), classifier.predict(inputTrain.get(i))))
          .count();
      ts = System.currentTimeMillis() - ts;
      System.out.printf("Training score: %f%% (%d out of %d); took %dms time\n", trainScore * 100D / trainSize,
          trainScore, trainSize, ts);
      counter.set(0);
      ts = System.currentTimeMillis();
      int testScore = (int) IntStream.range(0, testSize).unordered().parallel()
          .peek(i -> System.out.printf("%d entries checked out of %d\r", counter.incrementAndGet(), testSize))
          .filter(i -> Objects.equals(outcomeTest.get(i), classifier.predict(inputTest.get(i))))
          .count();
      ts = System.currentTimeMillis() - ts;
      System.out.printf("Test score: %f%% (%d out of %d); took %dms time\n", testScore * 100D / testSize, testScore,
          testSize, ts);
    }

    System.gc(); System.out.println();

    {
      var classifier = new KNearestNeighbors<Integer>(
          args.length > 2 ? Integer.parseInt(args[2]) : 5);

      System.out.println("KNN fitting starting...");
      long ts = System.currentTimeMillis();
      classifier.fit(inputTrain, outcomeTrain);
      ts = System.currentTimeMillis() - ts;
      System.out.println("Fitting took " + ts + "ms time");

      System.out.println("KNN evaluation starting...");
      final var counter = new java.util.concurrent.atomic.AtomicInteger(0);
      ts = System.currentTimeMillis();
      int trainScore = (int) IntStream.range(0, trainSize).unordered().parallel()
          .peek(i -> System.out.printf("%d entries checked out of %d\r", counter.incrementAndGet(), trainSize))
          .filter(i -> Objects.equals(outcomeTrain.get(i), classifier.predict(inputTrain.get(i))))
          .count();
      ts = System.currentTimeMillis() - ts;
      System.out.printf("Training score: %f%% (%d out of %d); took %dms time\n", trainScore * 100D / trainSize,
          trainScore, trainSize, ts);
      counter.set(0);
      ts = System.currentTimeMillis();
      int testScore = (int) IntStream.range(0, testSize).unordered().parallel()
          .peek(i -> System.out.printf("%d entries checked out of %d\r", counter.incrementAndGet(), testSize))
          .filter(i -> Objects.equals(outcomeTest.get(i), classifier.predict(inputTest.get(i))))
          .count();
      ts = System.currentTimeMillis() - ts;
      System.out.printf("Test score: %f%% (%d out of %d); took %dms time\n", testScore * 100D / testSize, testScore,
          testSize, ts);
    }
  }
}
