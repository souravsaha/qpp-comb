/*
 * Checks the status of the cluster; 
 * that is the number of terms of each of the clusters;
 */
package wordvectors;

import static common.CommonVariables.FIELD_WORD_VEC;
import java.io.File;
import java.io.IOException;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.apache.lucene.store.FSDirectory;



public class ClusterStatus {
    String clusterIndexPath;
    String indexPath;
    /**
     * The IndexReader that will be used for reading the index.
     */
    IndexReader clusterInfoReader;
    IndexSearcher clusterIdSearcher;

    public ClusterStatus() {
    }

    public ClusterStatus(String clusterIndexPath) throws IOException {

        this.clusterIndexPath = clusterIndexPath;

        clusterInfoReader = DirectoryReader.open(FSDirectory.open(new File(this.clusterIndexPath).toPath()));

        clusterIdSearcher = new IndexSearcher(clusterInfoReader);
        clusterIdSearcher.setSimilarity(new DefaultSimilarity());
        
    }

    public void showClusterStatus() throws IOException, ParseException {

        String cid = new String();
        int count = 0;
        int docCount = clusterInfoReader.maxDoc();      // total number of terms of the vocabulary that are clustered

        ScoreDoc[] hits = null;
        TopDocs topDocs = null;
        Query query;
        QueryParser clusterSearchParser = new QueryParser(FIELD_WORD_VEC, new WhitespaceAnalyzer());
        TopScoreDocCollector collector;

        while(count<docCount) {
            Document doc = clusterInfoReader.document(count);
            cid = doc.get(FIELD_WORD_VEC);
//            System.out.println("ClusterId: " + cid);

            collector = TopScoreDocCollector.create(docCount);
            query = clusterSearchParser.parse(cid);
            clusterIdSearcher.search(query, collector);

            topDocs = collector.topDocs();
            hits = topDocs.scoreDocs;
            int hits_length = hits.length;

            if(hits == null) {
                System.out.println("ClusterId not found for: " + cid);
                continue;
            }
            else {
                System.out.println(cid + " " + hits_length);
            }
            count += hits_length;
        }

    }

    public static void main(String[] args) throws IOException, ParseException {

        ClusterStatus clustStatus = new ClusterStatus("/wordvecsim_new/clusterids/100");

        clustStatus.showClusterStatus();
    }
    
}
