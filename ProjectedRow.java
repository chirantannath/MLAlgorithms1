import java.util.*;
final class ProjectedRow implements Row {
  private final Row innerRow;
  private final int[] indices;

  ProjectedRow(Row innerRow, int[] indices) {
    this.innerRow = innerRow; this.indices = indices;
  }
  @Override public Class<?>[] getColumnClasses() {
    return Arrays.stream(indices).mapToObj(innerRow::getColumnClass).toArray(Class<?>[]::new);
  }
  @Override public Class<?> getColumnClass(int index) throws IndexOutOfBoundsException {
    return innerRow.getColumnClass(indices[index]);
  }
  @Override public int getRowLength() {return indices.length;}
  @Override public Object get(int index) throws IndexOutOfBoundsException {return innerRow.get(indices[index]);}
  @Override public Optional<? extends Number> getAsNumber(int index) throws IndexOutOfBoundsException {return innerRow.getAsNumber(indices[index]);}
}