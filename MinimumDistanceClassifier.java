import java.util.*;
import java.util.function.*;

public class MinimumDistanceClassifier<C> implements Classifier<Float64Row, C> {
  /** Class means. */
  protected final Map<C, Float64RowStats> classMeans = new java.util.concurrent.ConcurrentHashMap<>();
  /** 
   * Parameter, distance function to be used, {@link Float64Row#distanceEuclidean(Float64Row)} 
   * by default.
   */
  public final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunction;

  public MinimumDistanceClassifier(ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc) {
    Objects.requireNonNull(distanceFunc); distanceFunction = distanceFunc;
  }
  public MinimumDistanceClassifier() {this(Float64Row::distanceEuclidean);}

  @Override public void fit(Float64Row input, C outputCls) {
    classMeans.merge(outputCls, new Float64RowStats(input, 1), (oldMean, initial) -> oldMean.add(input));
  }

  @Override public C predict(Float64Row input) {
    return classMeans.entrySet().stream()
    .min((e1, e2) -> Double.compare(
      distanceFunction.applyAsDouble(e1.getValue().getMean(), input),
      distanceFunction.applyAsDouble(e2.getValue().getMean(), input)
    ))
    .orElseThrow(IllegalStateException::new)
    .getKey();
  }

  public C predictParallel(Float64Row input) {
    return classMeans.entrySet().stream().unordered().parallel()
    .min((e1, e2) -> Double.compare(
      distanceFunction.applyAsDouble(e1.getValue().getMean(), input),
      distanceFunction.applyAsDouble(e2.getValue().getMean(), input)
    ))
    .orElseThrow(IllegalStateException::new)
    .getKey();
  }
}