import java.util.function.Consumer;
import java.util.stream.Collector;

/** Consumer to compute mean of Float64Row vectors. Mainly for internal use. */
public class Float64RowStats implements Consumer<Float64Row> {
  protected Float64Row sum;
  protected long count = 0;
  private Float64Row meanCached = null;

  public Float64RowStats(int numColumns) {sum = new Float64Row(numColumns);}
  public Float64RowStats(Float64Row initialSum, long initialCount) {
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
  public Float64RowStats add(Float64Row row) {
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

  public static Float64RowStats combine(Float64RowStats left, Float64RowStats right) {
    final int length = left.sum.getRowLength();
    if(length != right.sum.getRowLength())
      throw new IllegalArgumentException();
    final Float64RowStats result = new Float64RowStats(length);
    for(int i = 0; i < length; i++)
      result.sum.setAsDouble(i, left.sum.getAsDouble(i) + right.sum.getAsDouble(i));
    result.count = left.count + right.count;
    return result;
  }

  public static Collector<Float64Row, ?, Float64Row> meanCollector(final int rowLength) {
    return Collector.of(
      () -> new Float64RowStats(rowLength), 
      Float64RowStats::accept, 
      Float64RowStats::combine, 
      Float64RowStats::getMean,
      Collector.Characteristics.UNORDERED
    );
  }
  public static Collector<Float64Row, ?, Float64Row> sumCollector(final int rowLength) {
    return Collector.of(
      () -> new Float64RowStats(rowLength), 
      Float64RowStats::accept, 
      Float64RowStats::combine, 
      Float64RowStats::getSum,
      Collector.Characteristics.UNORDERED
    );
  }
}