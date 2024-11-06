/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import java.util.List;
import java.util.Locale;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.similarities.BasicStats;
import org.apache.lucene.search.similarities.LMSimilarity;


public class CustomSimilarity extends LMSimilarity {
    
  /** The &lambda; parameter. */
  private final float lambda;
  
  public CustomSimilarity() {
      lambda = 0;
  }

  /** Instantiates with the specified collectionModel and &lambda; parameter. */
  public CustomSimilarity(
      CollectionModel collectionModel, float lambda) {
    super(collectionModel);
    this.lambda = lambda;
  }

  /** Instantiates with the specified &lambda; parameter. */
  public CustomSimilarity(float lambda) {
    this.lambda = lambda;
  }
  

  @Override
  protected float score(BasicStats stats, float freq, float docLen) {
    return freq;
  }
  
  /** Returns the &lambda; parameter. */
  public float getLambda() {
    return lambda;
  }

  @Override
  public String getName() {
    return String.format(Locale.ROOT, "CustomSearcher_Returns_TF", getLambda());
  }

    @Override
    protected void explain(List<Explanation> subs, BasicStats stats, int doc,
        float freq, float docLen) {
    }
}
