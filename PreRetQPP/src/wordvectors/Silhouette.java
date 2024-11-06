/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wordvectors;

import java.util.List;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;


public class Silhouette {

    /**
     * silhouette = 
     * @param wordClusters
     * @return 
     */
    public static double returnSilhouette(List<CentroidCluster<WordVec>> wordClusters) {

        double ai;
        double silhouette = 0;
        int termCount = 0;

        // for each term of the list of terms:
        for (CentroidCluster<WordVec> cluster : wordClusters) {
            List<WordVec> clusterPoints = cluster.getPoints();

            // for calculating ai of each points of that cluster
            // for all terms of that cluster:
            for (WordVec clusterPoint1 : clusterPoints) {  // i = clusterPoint1
                // for each points of that cluster
                termCount++;
                ai = 0;
                int flag = 0;
                for (WordVec clusterPoint2 : clusterPoints) {
                    if(!clusterPoint1.isEqual(clusterPoint2)) {
                        flag = 1;
                        ai += clusterPoint1.euclideanDist(clusterPoint2);
                    }
                }
                if(flag == 1) // i.e. if the point is not the only member of the cluster
                    ai = ai / (double)(clusterPoints.size()-1); // -1 because distance with itself is avoided

                // for calculating bi
                // for each of the other clusters:
                double bi = 99999.0;
                for (CentroidCluster<WordVec> c : wordClusters) {
                    // calculate the distance with the cluster centroid, 
                    // other than its own, which has the minDistance with clusterPoint1
                    List<WordVec> cPoints = c.getPoints();
                    // if clusterPoint1 is *not* belongs to the this cluster c:
                    if (!cPoints.get(0).word.equals(clusterPoint1.word)) {
                        // then, calculate the distance between clusterPoint1 and c's centroid                        
                        double dist = clusterPoint1.euclideanDist(c.getCenter().getPoint());
                        if(dist < bi) {
                            bi = dist;
                        }                            
                    }
                }

                double s = (bi - ai)/Math.max(bi, ai);
                
                silhouette += s;
            }
        }

        System.out.println(silhouette / termCount);
        return silhouette / termCount;
    }

    public static float computeSilhouette(List<CentroidCluster<WordVec>> wordClusters) {
        int K = wordClusters.size();
        int i = 0, j = 0, k = 0;
        float a;
        float b;
        float silhouette;
        float avgSilhouette = 0;
        int M_k, M = 0;

        for (k=0; k < K; k++) {  // for each cluster

            CentroidCluster<WordVec> cluster = wordClusters.get(k);

            List<WordVec> clusterPoints = cluster.getPoints();
            M_k = clusterPoints.size();

            a = 0; b = 0;

            // Calculate a[i] for each i=1...M_k
            for (i=0; i<M_k; i++) {  // for each point in this cluster
                WordVec x = clusterPoints.get(i);

                // Compute a by averaging over distances between the members of its own cluster
                for (j=0; j<M_k; j++) {
                    if (i==j)
                        continue;

                    WordVec y = clusterPoints.get(j);
                    a += x.euclideanDist(y);
                }
                a = a / (float)M_k; // normalize


                // Compute b by averaging over distances between the this point and other centres
                for (int kprime=0; kprime < K; kprime++) {  // for each cluster
                    if (kprime == k)
                        continue;

                    CentroidCluster<WordVec> otherCluster = wordClusters.get(kprime);
                    Clusterable otherCentroid = otherCluster.getCenter();

                    b += x.euclideanDist(otherCluster.getCenter().getPoint()); 
                }
                
                b = b / (float)(K-1);

                silhouette = (b-a)/Math.max(a,b);
                avgSilhouette += silhouette;
            }
            M += M_k;
        }

        avgSilhouette /= (float)M;
//        System.out.println(avgSilhouette);
        return avgSilhouette;
    }
}
