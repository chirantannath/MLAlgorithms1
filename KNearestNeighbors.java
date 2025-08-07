import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;

public class KNearestNeighbors<C> implements Classifier<Float64Row, C> {
  /** Combined type to store input-output combination. */
  protected static record Pattern<CLS> (Float64Row input, CLS outputCls) {}
  /** Set of patterns collected till now. Usually modifiable and not thread-safe. */
  protected final List<Pattern<C>> knownPatterns = new ArrayList<>();
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
    knownPatterns.add(new Pattern<>(input, outputCls));
  }

  @Override public C predict(Float64Row input) {
    return knownPatterns.stream()//.unordered().parallel()
    .sorted((p1, p2) -> Double.compare(
      distanceFunction.applyAsDouble(p1.input, input),
      distanceFunction.applyAsDouble(p2.input, input)
    ))
    .limit(K)//.parallel()
    .collect(Collectors.groupingBy/*Concurrent*/(Pattern<C>::outputCls, Collectors.counting()))
    .entrySet().stream()//.unordered().parallel() //(K is assumed to be small)
    .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
    .orElseThrow(IllegalStateException::new)
    .getKey();
  }

  public C predictParallel(Float64Row input) {
    return knownPatterns.stream()/*.unordered()*/.parallel()
    .sorted((p1, p2) -> Double.compare(
      distanceFunction.applyAsDouble(p1.input, input),
      distanceFunction.applyAsDouble(p2.input, input)
    ))
    .limit(K).parallel()
    .collect(Collectors.groupingByConcurrent(Pattern<C>::outputCls, Collectors.counting()))
    .entrySet().stream()//.unordered().parallel() //(K is assumed to be small)
    .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
    .orElseThrow(IllegalStateException::new)
    .getKey();
  }
}
