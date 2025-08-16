import java.util.*;

final class ConcatenatedRow implements Row {
  private final Row[] rows;
  private final int[] cumulativeRowLengths;
  private final Class<?>[] columnClasses;

  ConcatenatedRow(Row[] rows) {
    this.rows = rows;
    cumulativeRowLengths = new int[rows.length];
    ArrayList<Class<?>> classes = new ArrayList<>();
    for(int i = 0; i < rows.length; i++) {
      classes.addAll(Arrays.asList(rows[i].getColumnClasses()));
      cumulativeRowLengths[i] = rows[i].getRowLength();
      if(i == 0) continue;
      cumulativeRowLengths[i] += cumulativeRowLengths[i - 1];
    }
    columnClasses = classes.toArray(new Class<?>[cumulativeRowLengths[rows.length-1]]);
  }

  @Override
  public Class<?>[] getColumnClasses() {return Arrays.copyOf(columnClasses, columnClasses.length);}

  @Override
  public Class<?> getColumnClass(int index) {return columnClasses[index];}

  @Override
  public int getRowLength() {return columnClasses.length;}

  @Override
  public Object get(int index) throws IndexOutOfBoundsException {
    final int rowIndex = Utils.higherBoundIndex(index, cumulativeRowLengths);
    return rows[rowIndex].get(index - cumulativeRowLengths[rowIndex]);
  }

  @Override
  public Optional<? extends Number> getAsNumber(int index) throws IndexOutOfBoundsException {
    final int rowIndex = Utils.higherBoundIndex(index, cumulativeRowLengths);
    return rows[rowIndex].getAsNumber(index - cumulativeRowLengths[rowIndex]);
  }

}
