import java.util.*;
import java.util.stream.IntStream;

/**
 * Transformer that rescales each attribute of the dataset to have a mean of 0
 * and a standard deviation of 1.
 */
public final class StandardScaler implements RowTransformer<Row, Float64Row> {
  /** Initial data points; this is cleared after fitting. */
  private List<Float64Row> data = new ArrayList<>();
  // Means and standard deviations; this is filled in after fitting.
  private double[] means = null, standardDeviations = null;
  /** Number of attributes this scaler is prepared for. */
  public final int rowLength;

  public StandardScaler(int rowLength) {
    if (rowLength <= 0)
      throw new IllegalArgumentException();
    this.rowLength = rowLength;
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
    final long N = data.parallelStream().unordered().count();
    if (N <= 0)
      throw new IllegalStateException("No data to fit");

    // Then the means
    means = IntStream.range(0, rowLength).parallel()
        .mapToDouble(column -> data.parallelStream().unordered().mapToDouble(row -> row.getAsDouble(column)).average()
            .orElseThrow(IllegalStateException::new))
        .toArray();

    // Then the standard deviations
    standardDeviations = IntStream.range(0, rowLength).parallel()
        .mapToDouble(column -> Math.sqrt(data.parallelStream().unordered().mapToDouble(row -> {
          final var step = row.getAsDouble(column) - means[column];
          return step * step / N;
        }).sum()))
        .toArray();

    data = null;
  }

  @Override
  public Float64Row transform(Row row) {
    final Float64Row result = new Float64Row(row);
    double stddev;
    for (int col = 0; col < rowLength; col++) {
      stddev = standardDeviations[col];
      result.setAsDouble(col, stddev == 0D ? 0D : (result.getAsDouble(col) - means[col]) / stddev);
    }
    return result;
  }
}
