import java.util.Optional;

/**
 * Represents a row of data.
 * 
 * @author chirantannath
 */
public interface Row {
  /**
   * Gets the class type for each column/attribute. At the same time can indicate
   * the length of the row. Must not return null.
   */
  Class<?>[] getColumnClasses();

  /** Gets the class type for one column. */
  default Class<?> getColumnClass(int index) throws IndexOutOfBoundsException {
    return getColumnClasses()[index];
  }

  /** Gets the length of the row. */
  default int getRowLength() {
    return getColumnClasses().length;
  }

  /** Gets an attribute. */
  Object get(int index) throws IndexOutOfBoundsException;

  /** Tries to get an attribute as a {@link Number}. */
  Optional<? extends Number> getAsNumber(int index) throws IndexOutOfBoundsException;

  /**
   * Gets a Row with only the columns specified. Columns <i>may be repeated</i>.
   * Data may either get deeply copied or it may be a view on the original row.
   * The default implementation returns a view on the original row.
   */
  default Row project(int... indices) throws IndexOutOfBoundsException {
    return new ProjectedRow(this, indices);
  }

  /**
   * Concatenates rows into a single Row. Subclasses are encouraged to define
   * their own versions of this static method. The default implementation returns
   * a view-like object over the given {@code rows}.
   */
  public static Row concatenate(Row... rows) {
    return new ConcatenatedRow(rows);
  }
}
