import java.io.*;
import java.util.*;
import java.util.stream.*;

public final class Int64Dataset extends Dataset<Int64Row>{
  public Int64Dataset(String[] columnNames, Int64Row[] data) 
    throws IllegalArgumentException, NullPointerException {
    super(columnNames, Int64Row.class, data);
  }
  public Int64Dataset(String[] columnNames, Stream<Int64Row> data) {
    super(columnNames, Int64Row.class, data);
  }
  public Int64Dataset(Dataset<? extends Int64Row> other) {
    super(other);
  }
  
  public long get(int row, int column) throws IndexOutOfBoundsException {
    return getDataRow(row).getAsLong(column);
  }
  public void set(int row, int column, long value) throws IndexOutOfBoundsException {
    getDataRow(row).setAsLong(column, value);
  }
  public synchronized void setSynchronized(int row, int column, long value) throws IndexOutOfBoundsException {
    getDataRow(row).setSynchronized(column, value);
  }

  public LongStream columnStreamAsLong(int column) throws IndexOutOfBoundsException {
    Objects.checkIndex(column, getNumColumns());
    return stream().mapToLong(row -> row.getAsLong(column));
  }
  @Override public Stream<Long> columnStream(int column) throws IndexOutOfBoundsException {
    return columnStreamAsLong(column).boxed();
  }
  
  @Override public Int64Dataset project(final int ... indices) throws IndexOutOfBoundsException {
    return new Int64Dataset(
      Arrays.stream(indices).mapToObj(this::getColumnName).toArray(String[]::new),
      stream().parallel().map(row -> row.project(indices))
    );
  }

  /**
   * Summarize all columns according to the order in {@link #getColumnNames()}.
   * This is done in parallel.
   */
  public LongSummaryStatistics[] summarize() {
    return IntStream.range(0, getNumColumns()).parallel()
    .mapToObj(col -> columnStreamAsLong(col).parallel().summaryStatistics())
    .toArray(LongSummaryStatistics[]::new);
  }

  public static Int64Dataset readCSV(String path) throws IOException {
    final var dataset = StringDataset.readCSV(path);
    try {
      final var datastream = dataset.stream().parallel().map(Int64Row::new);
      return new Int64Dataset(dataset.getColumnNames(), datastream);
    } catch (NoSuchElementException | IllegalArgumentException e) {
      throw new IOException(e);
    }
  }
}
