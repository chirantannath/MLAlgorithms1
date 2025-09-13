import java.util.*;

/**
 * Completely random classifier; during prediction chooses class labels at
 * random with all equal probability. To be used as a "control" experiment.
 * @author chirantannath
 */
public final class RandomClassifier<R extends Row, C> implements Classifier<R, C> {
  /** Random number source being used. */
  public final Random rng;
  /** What classes have been seen? */
  private Set<C> classes = new HashSet<>();
  /** Set when all classes have been seen. */
  private Object[] allClasses = null;

  public RandomClassifier(Random rng) {
    this.rng = Objects.requireNonNull(rng);
  }

  public RandomClassifier() {
    this(new Random());
  }

  @Override
  public void fit(R input, C trueOutput) {
    classes.add(trueOutput);
  }

  @Override
  public void finishFitting() {
    allClasses = classes.toArray();
    classes = null;
  }

  @Override
  @SuppressWarnings("unchecked")
  public Optional<C> predict(R input) {
    return Optional.of((C) allClasses[rng.nextInt(allClasses.length)]);
  }
}
