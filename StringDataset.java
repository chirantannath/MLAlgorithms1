import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.*;

public final class StringDataset extends Dataset<StringRow> {
  public StringDataset(String[] columnNames, StringRow[] data) 
    throws IllegalArgumentException, NullPointerException {
    super(columnNames, StringRow.class, data);
  }
  public StringDataset(String[] columnNames, Stream<StringRow> data) {
    super(columnNames, StringRow.class, data);
  }
  /** Convert any other dataset into StringDataset by calling {@link Object#toString()}. */
  public <R extends Row> StringDataset(Dataset<R> other) {
    super(other.getColumnNames(), StringRow.class, other.stream().parallel().map(StringRow::new));
  }

  public String get(int row, int column) throws IndexOutOfBoundsException {
    return getDataRow(row).get(column);
  }
  public void set(int row, int column, String value) throws IndexOutOfBoundsException {
    getDataRow(row).set(column, value);
  }
  public synchronized void setSynchronized(int row, int column, String value) throws IndexOutOfBoundsException {
    getDataRow(row).setSynchronized(column, value);
  }

  @Override public Stream<String> columnStream(int column) throws IndexOutOfBoundsException {
    Objects.checkIndex(column, getNumColumns());
    return stream().map(row -> row.get(column));
  }

  @Override public StringDataset project(final int ... indices) throws IndexOutOfBoundsException {
    return new StringDataset(
      Arrays.stream(indices).mapToObj(this::getColumnName).toArray(String[]::new),
      stream().parallel().map(row -> row.project(indices))
    );
  }

  public static final Pattern CSV_SEPARATOR = Pattern.compile("\\,");

  public static StringDataset readCSV(String path) throws IOException {
    String[] columnNames;
    Stream.Builder<String[]> data = Stream.builder();
    try (Scanner scanner = new Scanner(new File(path))) {
      //The first line is the columns
      columnNames = CSV_SEPARATOR.split(scanner.nextLine());
      int lineNumber = 1;
      String line;
      while(scanner.hasNextLine()) {
        lineNumber++;
        var row = CSV_SEPARATOR.split(line = scanner.nextLine());
        if(row.length != columnNames.length)
          throw new IOException("column number mismatch at line "+lineNumber+"\nfor line \'"+line+"\'");
        data.add(row);
      }
      return new StringDataset(columnNames, data.build().parallel().map(StringRow::new));
    } catch (NoSuchElementException e) {
      throw new IOException(e);
    }
  }
}
