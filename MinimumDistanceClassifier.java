import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class MinimumDistanceClassifier<C> implements Classifier<Float64Row, C> {
  /** Consumer to compute mean of Float64Row vectors. */
  protected static class RowMean implements Consumer<Float64Row> {
    protected Float64Row sum;
    protected long count = 0;
    private Float64Row meanCached = null;
  
    public RowMean(int numColumns) {sum = new Float64Row(numColumns);}
    public RowMean(Float64Row initialSum, long initialCount) {
      sum = new Float64Row(initialSum); count = initialCount;
    }

    @Override public void accept(Float64Row row) {
      meanCached = null;
      final int length = sum.getRowLength();
      if(row.getRowLength() != length)
        throw new IllegalArgumentException("mismatched column length");
      for(int i = 0; i < length; i++)
        sum.setAsDouble(i, sum.getAsDouble(i) + row.getAsDouble(i));
      count++;
    }
    public RowMean add(Float64Row row) {
      meanCached = null;
      accept(row); return this;
    }

    public Float64Row getMean() {
      if(meanCached != null) return meanCached;
      final int length = sum.getRowLength();
      final Float64Row result = new Float64Row(length);
      for(int i = 0; i < length; i++)
        result.setAsDouble(i, sum.getAsDouble(i)/count);
      return meanCached = result;
    }
    public Float64Row getSum() {return new Float64Row(sum);}
    public long getCount() {return count;}

    public static RowMean combine(RowMean left, RowMean right) {
      final int length = left.sum.getRowLength();
      if(length != right.sum.getRowLength())
        throw new IllegalArgumentException();
      final RowMean result = new RowMean(length);
      for(int i = 0; i < length; i++)
        result.sum.setAsDouble(i, left.sum.getAsDouble(i) + right.sum.getAsDouble(i));
      result.count = left.count + right.count;
      return result;
    }

    public static Collector<Float64Row, ?, Float64Row> meanCollector(final int rowLength) {
      return Collector.of(
        () -> new RowMean(rowLength), 
        RowMean::accept, 
        RowMean::combine, 
        RowMean::getMean,
        Collector.Characteristics.UNORDERED
      );
    }
    public static Collector<Float64Row, ?, Float64Row> sumCollector(final int rowLength) {
      return Collector.of(
        () -> new RowMean(rowLength), 
        RowMean::accept, 
        RowMean::combine, 
        RowMean::getSum,
        Collector.Characteristics.UNORDERED
      );
    }
  }

  /** Class means. */
  protected final Map<C, RowMean> classMeans = new java.util.concurrent.ConcurrentHashMap<>();
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
    classMeans.merge(outputCls, new RowMean(input, 1), (c, mean) -> mean.add(input));
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