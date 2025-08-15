import java.util.*;
import java.util.stream.IntStream;

/**
 * Transformer that rescales each attribute of the dataset based on the first,
 * second and third quartiles of those values and the interquartile range (IQR).
 * This is supposed to be <strong>robust</strong> against outlier values.
 */
public final class RobustScaler implements RowTransformer<Row, Float64Row> {
  /** Initial data points; this is cleared after fitting. */
  private List<Float64Row> data = new ArrayList<>();
  // The first, second and third quartiles, this is filled in after fitting.
  private final double[] firstQuartiles, secondQuartiles, thirdQuartiles;
  /** Number of attributes this scaler was prepared for. */
  public final int rowLength;

  public RobustScaler(int rowLength) {
    if (rowLength <= 0)
      throw new IllegalArgumentException();
    this.rowLength = rowLength;
    firstQuartiles = new double[rowLength];
    secondQuartiles = new double[rowLength];
    thirdQuartiles = new double[rowLength];
  }

  @Override
  public void fit(Row row) {
    if (row.getRowLength() != rowLength)
      throw new IllegalArgumentException();
    data.add(new Float64Row(row));
  }

  @Override
  public void finishFitting() {
    if (data == null)
      return;

    // First the count
    final int N = (int) data.parallelStream().unordered().count();
    if (N <= 0)
      throw new IllegalStateException("No data to fit");

    IntStream.range(0, rowLength).unordered().parallel()
        .forEach(col -> {
          final double[] colArray = data.parallelStream()
              .mapToDouble(row -> row.getAsDouble(col))
              .parallel().sorted().parallel().toArray();
          firstQuartiles[col] = firstQuartile(colArray, 0, N);
          secondQuartiles[col] = median(colArray, 0, N);
          thirdQuartiles[col] = thirdQuartile(colArray, 0, N);
        });

    data = null;
  }

  @Override
  public Float64Row transform(Row row) {
    final Float64Row result = new Float64Row(row);
    double colSpanRange;
    for (int col = 0; col < rowLength; col++) {
      colSpanRange = thirdQuartiles[col] - firstQuartiles[col];
      if (colSpanRange == 0)
        colSpanRange = 1D;
      result.setAsDouble(col, (result.getAsDouble(col) - secondQuartiles[col]) / colSpanRange);
    }
    return result;
  }

  /** Returns the median, assumes sorted array. */
  public static double median(double[] array, int startInclusive, int endExclusive) {
    Objects.checkFromToIndex(startInclusive, endExclusive, array.length);
    final int N = endExclusive - startInclusive;
    if (N <= 0)
      throw new IllegalArgumentException();
    final int midIdx = startInclusive + (N >>> 1);
    final double midValue2 = array[midIdx];
    return (N & 1) == 0 ? Math.scalb(array[midIdx - 1] + midValue2, -1) : midValue2;
  }

  /** Returns the first quartile, assumes sorted array. */
  public static double firstQuartile(double[] array, int startInclusive, int endExclusive) {
    final int N = endExclusive - startInclusive;
    return N == 1 ? array[startInclusive] : median(array, startInclusive, startInclusive + (N >>> 1));
  }

  /** Returns the third quartile, assumes sorted array. */
  public static double thirdQuartile(double[] array, int startInclusive, int endExclusive) {
    final int N = endExclusive - startInclusive;
    return N == 1 ? array[startInclusive] : median(array, startInclusive + ((N + 1) >>> 1), endExclusive);
  }
}
