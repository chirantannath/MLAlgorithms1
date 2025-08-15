import java.util.*;

/**
 * Transformer that rescales each attribute of the dataset from the minimum and
 * maximum to a different range.
 */
public final class MinMaxScaler implements RowTransformer<Row, Float64Row> {
  /** Minimums of each attribute. */
  private final double[] minimums;
  /** Maximums of each attribute. */
  private final double[] maximums;
  /** Target minimum, inclusive. */
  public final double targetMinimum;
  /** Target maximum, inclusive. */
  public final double targetMaximum;
  /** Number of attributes this scaler is prepared for. */
  public final int rowLength;

  public MinMaxScaler(double targetMinimum, double targetMaximum, int rowLength) {
    if (!Double.isFinite(targetMinimum))
      throw new IllegalArgumentException("targetMinimum");
    this.targetMinimum = targetMinimum;
    if (!Double.isFinite(targetMaximum))
      throw new IllegalArgumentException("targetMaximum");
    this.targetMaximum = targetMaximum;
    if (rowLength <= 0)
      throw new IllegalArgumentException("rowLength");
    this.rowLength = rowLength;
    minimums = new double[rowLength];
    Arrays.fill(minimums, Double.POSITIVE_INFINITY);
    maximums = new double[rowLength];
    Arrays.fill(maximums, Double.NEGATIVE_INFINITY);
  }

  public MinMaxScaler(int rowLength) {
    this(0, 1, rowLength);
  }

  @Override
  public void fit(Row row) {
    if (row.getRowLength() != rowLength)
      throw new IllegalArgumentException();
    double val;
    final var numberRow = new Float64Row(row);
    for (int col = 0; col < rowLength; col++) {
      val = numberRow.getAsDouble(col);
      if (val > maximums[col])
        maximums[col] = val;
      if (val < minimums[col])
        minimums[col] = val;
    }
  }

  @Override
  public Float64Row transform(Row row) {
    final var spanRange = targetMaximum - targetMinimum;
    final var result = new Float64Row(row);
    double colSpanRange;
    for (int col = 0; col < rowLength; col++) {
      colSpanRange = maximums[col] - minimums[col];
      result.setAsDouble(col, colSpanRange == 0 || spanRange == 0 ? targetMinimum
          : (result.getAsDouble(col) - minimums[0]) * spanRange / colSpanRange + targetMinimum);
    }
    return result;
  }

}
