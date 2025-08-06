import java.util.*;
import java.util.stream.*;

final class ClassifierTest {
  private ClassifierTest() {}

  public static void main(String args[]) throws Exception {
    final var original = StringDataset.readCSV(args.length > 0 ? args[0] : "diabetes.csv");
    System.out.println("Number of rows: "+original.size());
    System.out.println("Columns: "+Arrays.toString(original.getColumnNames()));

    final Random rng = new Random(12012001L); //my birthday as my seed
    final StringDataset originalTrain, originalTest;
    final double splitFraction = args.length > 1 ? Double.parseDouble(args[1]) : 0.75;
    {
      final var split = Dataset.split(original, rng, splitFraction).parallel()
      .map(p -> new StringDataset(original.getColumnNames(), p.parallel()))
      .collect(Collectors.toUnmodifiableList());
      assert split.size() == 2;
      originalTrain = split.get(0); originalTest = split.get(1);
    }
    final int trainSize = originalTrain.size(), testSize = originalTest.size();
    System.out.printf("Training size: %d, Test size: %d\n", trainSize, testSize);
    assert trainSize + testSize == original.size();

    //Input feature indices
    final int[] inFeatures = IntStream.range(0, original.getNumColumns()-1).toArray();
    final List<Integer> outcomeTrain = originalTrain.columnStream(original.getNumColumns()-1).parallel()
    .map(String::trim).map(Integer::valueOf).collect(Collectors.toUnmodifiableList());
    final List<Integer> outcomeTest = originalTest.columnStream(original.getNumColumns()-1).parallel()
    .map(String::trim).map(Integer::valueOf).collect(Collectors.toUnmodifiableList());
    final List<Float64Row> inputTrain = originalTrain.stream().parallel()
    .map(row -> new Float64Row(row.project(inFeatures)))
    .collect(Collectors.toUnmodifiableList());
    final List<Float64Row> inputTest = originalTest.stream().parallel()
    .map(row -> new Float64Row(row.project(inFeatures)))
    .collect(Collectors.toUnmodifiableList());

    final var classifier = new KNearestNeighbors<Integer>(args.length > 2 ? Integer.parseInt(args[2]) : 5);
    classifier.fit(inputTrain, outcomeTrain);

    System.out.println("KNN evaluation starting...");
    int trainScore = classifier.countCorrect(inputTrain, outcomeTrain);
    System.out.printf("Training score: %f%% (%d out of %d)\n", trainScore * 100D / trainSize, trainScore, trainSize);
    int testScore = classifier.countCorrect(inputTest, outcomeTest);
    System.out.printf("Test score: %f%% (%d out of %d)\n", testScore * 100D / testSize, testScore, testSize);
  }
}
