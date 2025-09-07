import java.util.*;
import java.util.function.*;
//import java.util.stream.*;

public class KNearestNeighbors<C> implements Classifier<Float64Row, C> {
  /** Set of patterns collected till now. Usually modifiable and not thread-safe. */
  protected final List<Pair<Float64Row, C>> knownPatterns = new ArrayList<>();
  /** Parameter, number of nearest neighbors selected for prediction */
  public final int K;
  /** 
   * Parameter, distance function to be used, {@link Float64Row#distanceEuclidean(Float64Row)} 
   * by default.
   */
  public final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunction;

  public KNearestNeighbors(final int k, final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc) {
    if(k <= 0) throw new IllegalArgumentException();
    K = k;
    Objects.requireNonNull(distanceFunc);
    distanceFunction = distanceFunc;
  }
  public KNearestNeighbors(final int k) {
    this(k, Float64Row::distanceEuclidean);
  }

  @Override public void fit(Float64Row input, C outputCls) {
    knownPatterns.add(new Pair<>(input, outputCls));
  }

  @Override public Optional<C> predict(Float64Row input) {
    final var selected = Utils.selectSmallestK(knownPatterns.iterator(), (p1, p2) -> Double.compare(
       distanceFunction.applyAsDouble(p1.first(), input),
       distanceFunction.applyAsDouble(p2.first(), input)
    ), K);
    final var counts = new HashMap<C, Integer>();
    for(var p : selected) 
      counts.put(p.second(), counts.getOrDefault(p.second(), 0) + 1);
    if(counts.isEmpty()) throw new IllegalStateException("Nothing fitted in");
    C maxClass = null; int maxClassCount = 0;
    for(var entry : counts.entrySet()) {
      final int classCount = entry.getValue();
      if(classCount > maxClassCount) {
        maxClass = entry.getKey(); 
        maxClassCount = classCount;
      }
    }
    return Optional.of(maxClass);
  }
}
