import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** 
 * Dataset transformation implementation that transforms {@link Row}s in some way
 * (after <i>learning</i> from the characteristics of a whole dataset at once).
 */
public interface RowTransformer<IR extends Row, OR extends Row> {
  /** 
   * Inform this transformer about a new dataset row. This changes state of this
   * object so it should be called in a thread-safe manner.
   */
  void fit(IR row);
  /** Inform this transformer about a new sequence of dataset rows. */
  default void fit(Iterator<? extends IR> rows) {
    while(rows.hasNext())
      fit(rows.next());
  }
  /** 
   * This is called for final post-processing work after all fitting; model should
   * not change state after this method returns.
   * 
   * <p>The default implementation does nothing.</p>
   */
  default void finishFitting() {}

  /** Transform the given input row. */
  OR transform(IR row);
  /** Transforms the given sequence of rows. */
  default Stream<? extends OR> transform(Stream<? extends IR> rows) {
    return rows.map(this::transform);
  }

  /** Do fitting and transformation in one move. */
  default List<? extends OR> fitAndTransform(List<? extends IR> rows) {
    fit(rows.iterator());
    finishFitting();
    return transform(rows.parallelStream()).collect(Collectors.toUnmodifiableList());
  }
}
