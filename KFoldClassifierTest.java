import java.util.*;
import java.util.stream.*;

final class KFoldClassifierTest {
  private KFoldClassifierTest() {
  }
  
  public static void main(String args[]) throws Exception {
    final List<Pair<Float64Row, String>> dataset;
    final String[] inputColumnNames;
    
    try (final Scanner sc = new Scanner(System.in)) {
      //Prepare dataset from input file
      {
        
      
      }    

      System.gc();
      System.out.println();

      
    }
  }
  
  static <IR extends Row, OP> void testClassifier(
      Classifier<IR, OP> classifier,
      Stream<Pair<IR, OP>> trainingDataStream,
      Stream<Pair<IR, OP>> testingDataStream) {
    final var trainingData = trainingDataStream.parallel().collect(Collectors.toCollection(ArrayList::new));
    final var testingData = testingDataStream.collect(Collectors.toCollection(ArrayList::new));
    final var trainSize = trainingData.size();
    final var testSize = testingData.size();
    
    System.out.println(classifier.getClass().getName() + " fitting starting...");
    long ts = System.currentTimeMillis();
    classifier.fit(trainingData.iterator());
    classifier.finishFitting();
    ts = System.currentTimeMillis() - ts;
    System.out.println("Fitting took " + ts + "ms time");
    
    System.out.println(classifier.getClass().getName() + " evaluation starting...");

    ts = System.currentTimeMillis();
    var trainScore = classifier.countCorrectParallel(trainingData.parallelStream().unordered(), null);
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Training accuracy score: %f%% (%d out of %d); took %dms time\n", trainScore * 100D / trainSize,
        trainScore, trainSize, ts);
        
    ts = System.currentTimeMillis();
    var testScore = classifier.countCorrectParallel(testingData.parallelStream().unordered(), null);
    ts = System.currentTimeMillis() - ts;
    System.out.printf("Testing accuracy score: %f%% (%d out of %d); took %dms time\n", testScore * 100D / testSize,
        testScore, testSize, ts);  
  }
}
