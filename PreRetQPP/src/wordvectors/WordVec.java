
package wordvectors;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.lucene.util.BytesRef;


public class WordVec implements Comparable<WordVec>, Serializable, Clusterable {

    String  word;       // the word
    float[] vec;        // the vector for that word
    float   norm;       // the normalization factor
    float   querySim;   // distance from a reference query point
    transient boolean isComposed; // whether it is composed word or not
    int     clusterId;  // the cluster-id corresponding to the word, if associated

    public static final String COMPOSING_DELIM = ":";
    
    public WordVec(int dimension) { vec = new float[dimension]; }

    /**
     * Initialize Word and corresponding Vector from 'line'
     * @param line Containing word and vector in a line; first token is word; 
     * rest are vector components.
     */
    public WordVec(String line) {
        String[] tokens = line.split("\\s+");
        word = tokens[0];
        vec = new float[tokens.length-1];
        for (int i = 1; i < tokens.length; i++)
            vec[i-1] = Float.parseFloat(tokens[i]);
    }

    /**
     * Initialize Word and corresponding Vector
     * @param word The word
     * @param vec Corresponding vector
     */
    public WordVec(String word, double[] vec) {
        this.word = word;
        float[] fvec = new float[vec.length];
        for (int i=0; i<vec.length; i++)
            fvec[i] = (float)vec[i];
        
        this.vec = fvec;
    }
    

    /**
     * Normalize components in [0, 1] The wordvecs are in [-1, 1];
     * I am not sure about the actual range to be in [-1, 1];
     * I think it is between [-2, 2].
     */
    public void normalize() {
        for (int i=0; i<vec.length; i++) {
            vec[i] = (vec[i]+1)/2;
        }
    }

    /**
     * Whether the vector stored, is a composed vector or not.
     * @return TRUE if composed; else FALSE.
     */
    public boolean isComposed() { return isComposed; }

    /**
     * Whether the two wordVec are same or not
     * @param that
     * @return true, if the WordVec are equal
     */
    public boolean isEqual(WordVec that) { return this.word.equals(that.word); }

    /**
     * Sets the cluster-id of the word using the vocabulary cluster.
     * @param clusterId The corresponding cluster-id.
     */
    public void setClusterId(int clusterId) { this.clusterId = clusterId; }

    /**
     * Returns the cluster-id associated with the word.
     * @return The associated cluster-id.
     */
    public int getClusterId() { return clusterId; }

    public void setNorm() {
        if (norm > 0)
            return;
        
        // calculate and store
        float sum = 0;
        for (int i = 0; i < vec.length; i++) {
            sum += vec[i]*vec[i];
        }
        norm = (float)Math.sqrt(sum);
    }

    /**
     * Returns the normalization factor of the vector.
     * @return The normalization factor.
     */
    public float getNorm() {
        if (norm > 0)
            return norm;
        
        setNorm();
        return norm;
    }

    /**
     * Returns the dimension of the vector.
     * @return The dimension of the vector.
     */
    public int getDimension() { return this.vec.length; }

    /**
     * Returns the centroid (or, sum) vector of two vectors.
     * @param a The vector to be summed
     * @param b The vector to be summed
     * @return The centroid (or, sum) of vectors a and b
     */
    static public WordVec centroid(WordVec a, WordVec b) {
        WordVec sum = new WordVec(a.vec.length);
        sum.word = a.word + COMPOSING_DELIM + b.word;
        for (int i = 0; i < a.vec.length; i++) {
            sum.vec[i] = (a.vec[i] + b.vec[i]);
        }
        sum.isComposed = true;
        return sum;
    }

    public static WordVec add(WordVec a, WordVec b) {
        WordVec sum = new WordVec(a.word + ":" + b.word);
        sum.vec = new float[a.vec.length];
        for (int i = 0; i < a.vec.length; i++) {
            sum.vec[i] = .5f * (a.vec[i]/a.getNorm() + b.vec[i]/b.getNorm());
        }        
        sum.isComposed = true;
        return sum;
    }


    /**
     * Returns the cosine similarity between the two vectors.
     * @param that Second vector to compute the cosine similarity.
     * @return The cosine similarity between 'this' and 'that'.
     */
    public float cosineSim(WordVec that) {
        float sum = 0;
        for (int i = 0; i < this.vec.length; i++) {
            sum += vec[i] * that.vec[i];
        }
        return sum/(this.getNorm()*that.getNorm());
    }

    /**
     * Returns the Euclidean distance between the two vectors.
     * @param that Second vector to compute the cosine similarity.
     * @return The Euclidean distance between 'this' and 'that'.
     */
    public float euclideanDist(WordVec that) {
        float sum = 0;
        for (int i = 0; i < this.vec.length; i++) {
            sum += (vec[i] - that.vec[i]) * ((vec[i] - that.vec[i]));
        }
        return (float)Math.sqrt(sum);
    }

    public float euclideanDist(double v[]) {
        float sum = 0;
        for (int i = 0; i < this.vec.length; i++) {
            sum += (this.vec[i] - v[i]) * ((this.vec[i] - v[i]));
        }
        return (float)Math.sqrt(sum);
    }

    /**
     * Returns the associated word
     * @return The associated word
     */
    public String getWord() { return word; }
    /**
     * Set the word
     * @param name The word 
     */
    public void setWord(String name) { this.word = name; }
    
    public float getQuerySim() { return querySim; }
    
    @Override
    public int compareTo(WordVec that) {
        return this.querySim > that.querySim? -1 : this.querySim == that.querySim? 0 : 1;
    }
    
    public byte[] getBytes() throws IOException {
        byte[] byteArray;
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            ObjectOutput out;
            out = new ObjectOutputStream(bos);
            out.writeObject(this);
            byteArray = bos.toByteArray();
            out.close();
        }
        return byteArray;
    }
    
    static WordVec decodeFromByteArray(BytesRef bytes) throws Exception {
        ObjectInput in;
        Object o;
        try (ByteArrayInputStream bis = new ByteArrayInputStream(bytes.bytes)) {
            in = new ObjectInputStream(bis);
            o = in.readObject();
        }
        in.close();
        return (WordVec)o;
    }

    @Override
    public double[] getPoint() {
        double[] vec = new double[this.vec.length];
        for (int i=0; i<this.vec.length; i++)
            vec[i] = this.vec[i];
        return vec;
    }
    
    @Override
    public String toString() {
        StringBuffer buff = new StringBuffer(word);

        buff.append(" ");
        if(this.vec != null) {
            for (double d : this.vec) {
                buff.append(d).append(" ");
            }
        }
        else
            buff.append("No-vector");
            
        return buff.toString();
    }
}

