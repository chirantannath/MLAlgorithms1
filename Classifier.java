import java.util.*;
import java.util.function.*;
import java.util.stream.*;
/**
 * Interface for classifiers (predicting discrete output from input).
 * @param <IR> Input row type on which to classify
 * @param <OP> Output classificiation result type
 * @author chirantannath
 */
public interface Classifier<IR extends Row, OP> {
  /** Train (further) on a <i>single</i> input-output combination. */
  void fit(IR input, OP trueOutput);
  /** '
   * Train (further) on a set of input-output combinations. 
   */
  default void fit(List<? extends IR> inputs, List<? extends OP> trueOutputs) {
    final int size = inputs.size();
    if(size != trueOutputs.size()) throw new IllegalArgumentException();
    for(int i = 0; i < size; i++) 
      fit(inputs.get(i), trueOutputs.get(i));
  }
  /** Predict for an input. This NEEDS to be thread-safe. */
  OP predict(IR input);
  /** 
   * (Lazily) predict for a set of inputs.
   * Default implementation behaves as if
   * <pre>
   * return inputs.map(this::predict);
   * </pre>
   * @see #predict(IR)
   */
  default Stream<? extends OP> predict(Stream<? extends IR> inputs) {
    return inputs.map(this::predict);
  }
  /** Count correctly predicted. (This is not a lazy operation.) */
  default int countCorrect(List<? extends IR> inputs, List<? extends OP> trueOutputs) {
    final int size = inputs.size();
    if(size != trueOutputs.size()) throw new IllegalArgumentException();
    return (int)IntStream.range(0, size)//.parallel() (Not parallel by default)
    .filter(i -> Objects.equals(trueOutputs.get(i), predict(inputs.get(i))))
    .count();
  }
  default int countCorrectParallel(List<? extends IR> inputs, List<? extends OP> trueOutputs) {
    final int size = inputs.size();
    if(size != trueOutputs.size()) throw new IllegalArgumentException();
    return (int)IntStream.range(0, size).unordered().parallel()
    .filter(i -> Objects.equals(trueOutputs.get(i), predict(inputs.get(i))))
    .count();
  }
  
  public static <IR extends Row, OP> Classifier<IR, OP> of(BiConsumer<IR, OP> fitter, Function<IR, OP> predictor) {
    return new Classifier<IR, OP>() {
      @Override public void fit(IR input, OP trueOutput) {fitter.accept(input, trueOutput);}
      @Override public OP predict(IR input) {return predictor.apply(input);}
    };
  }
}
