import java.io.*;
import java.util.*;
import java.util.stream.*;

public final class Float64Dataset extends Dataset<Float64Row> {
  public Float64Dataset(String[] columnNames, Float64Row[] data)
      throws IllegalArgumentException, NullPointerException {
    super(columnNames, Float64Row.class, data);
  }

  public Float64Dataset(String[] columnNames, Stream<Float64Row> data) {
    super(columnNames, Float64Row.class, data);
  }

  public Float64Dataset(Float64Dataset other) {
    super(other);
  }

  public Float64Dataset(Dataset<? extends Row> other) {
    this(other.getColumnNames(), other.stream().parallel().map(Float64Row::new));
  }

  public double get(int row, int column) throws IndexOutOfBoundsException {
    return getDataRow(row).getAsDouble(column);
  }

  public void set(int row, int column, double value) throws IndexOutOfBoundsException {
    getDataRow(row).setAsDouble(column, value);
  }

  public synchronized void setSynchronized(int row, int column, double value) throws IndexOutOfBoundsException {
    getDataRow(row).setSynchronized(column, value);
  }

  @Override
  public Stream<Double> columnStream(int column) throws IndexOutOfBoundsException {
    return columnStreamAsDouble(column).boxed();
  }

  public DoubleStream columnStreamAsDouble(int column) throws IndexOutOfBoundsException {
    Objects.checkIndex(column, getNumColumns());
    return stream().mapToDouble(row -> row.getAsDouble(column));
  }

  @Override
  public Float64Dataset project(final int... indices) throws IndexOutOfBoundsException {
    return new Float64Dataset(
        Arrays.stream(indices).mapToObj(this::getColumnName).toArray(String[]::new),
        stream().parallel().map(row -> row.project(indices)));
  }

  /**
   * Summarize all columns according to the order in {@link #getColumnNames()}.
   * This is done in parallel.
   */
  public DoubleSummaryStatistics[] summarize() {
    return IntStream.range(0, getNumColumns()).parallel()
        .mapToObj(col -> columnStreamAsDouble(col).parallel().summaryStatistics())
        .toArray(DoubleSummaryStatistics[]::new);
  }

  public static Float64Dataset readCSV(String path) throws IOException {
    final var dataset = StringDataset.readCSV(path);
    try {
      return new Float64Dataset(dataset);
    } catch (NoSuchElementException | IllegalArgumentException e) {
      throw new IOException(e);
    }
  }

  public static void inplaceFitAndTransform(List<Float64Row> rows,
      RowTransformer<? super Row, ? extends Float64Row> transformer) {
    transformer.fit(rows.iterator());
    transformer.finishFitting();
    inplaceTransform(rows, transformer);
  }
  public static void inplaceFitAndTransformParallel(List<Float64Row> rows,
      RowTransformer<? super Row, ? extends Float64Row> transformer) {
    transformer.fit(rows.iterator());
    transformer.finishFitting();
    inplaceTransformParallel(rows, transformer);
  }

  public static void inplaceTransform(List<Float64Row> rows,
      RowTransformer<? super Row, ? extends Float64Row> transformer) {
    final var itr = rows.listIterator();
    while(itr.hasNext()) {
      itr.set(transformer.transform(itr.next()));
    }
  }
  public static void inplaceTransformParallel(List<Float64Row> rows,
      RowTransformer<? super Row, ? extends Float64Row> transformer) {
    IntStream.range(0, rows.size()).unordered().parallel()
        .forEach(idx -> rows.set(idx, transformer.transform(rows.get(idx))));
  }
}
