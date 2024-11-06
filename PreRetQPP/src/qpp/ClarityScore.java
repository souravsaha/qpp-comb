/**
 * Predicting Query Performance
 * Steve Cronen-Townsend, Yun Zhou, Bruce Croft
 * In SIGIR 2002.
 */
package qpp;

import common.TRECQuery;
import common.TRECQueryParser;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.store.FSDirectory;
import org.xml.sax.SAXException;



public class ClarityScore {

    Properties      prop;
    IndexReader     indexReader;
    IndexSearcher   searcher;
    String          indexPath;
    String          queryPath;
    TrecDocAnalyzer trecAnalyzer;
    Analyzer        analyzer;
    String          runName;
    int             numWanted;
    CollectionStatistics collStat;
    float           lambda;         // linear smoothing parameter
    List<TRECQuery> queries;
    TRECQueryParser trecQueryparser;
    String fieldToSearch;

    public ClarityScore(String propPath) throws IOException, SAXException, Exception {
        prop = new Properties();
        prop.load(new FileReader(propPath));

        indexPath = prop.getProperty("indexPath");
        File indexDir;
        indexDir = new File(indexPath);
        indexReader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        searcher = new IndexSearcher(indexReader);

        trecAnalyzer = new TrecDocAnalyzer(prop);
        trecAnalyzer.setAnalyzer();
        analyzer = trecAnalyzer.getAnalyzer();

        runName = "qpp-initial-ret";
        numWanted = 500;    // set to 500 following the paper of clarity score

        collStat = new CollectionStatistics(indexPath, "content");
        collStat.buildCollectionStat();

        lambda = 0.6f;   // lambda set to 0.6 as done in Clarity score

        /* setting query path */
        String queryPath = prop.getProperty("queryPath");
        /* constructing the query */
        fieldToSearch = "wordname";
        trecQueryparser = new TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = trecQueryparser.constructQueries();
        /* constructed the query */


    }

    /**
     * Returns the MLE of 'term' in document with lambda linear smoothing parameter
     * @param w
     * @param dv
     * @param lambda
     * @return 
     * MLE(w, D, lambda) = lambda*tf(w,D)/|D| + (1-lambda)*cf(w)/|coll|
     */
    public double getMLE(String w, DocumentVector dv, double lambda) {
        double mle = 0;
        PerTermStat pts = dv.docVec.get(w);

        mle = lambda * ( (pts==null)?0:pts.rcf ) + 
                        (1-lambda) * collStat.perTermStat.get(w).rcf;

        return mle;
    }

    public void retrieveAll() throws Exception {

        ScoreDoc[] hits = null;
        TopDocs topDocs = null;
        FileWriter resFileWriter = new FileWriter("trec8.xml-LMJelinek-Mercer0.6-D20-T70-rm3-0.6.res");

        //* sum of relative collection frequency of all terms 'w' in the collection as a whole
        double sum_rcf_V = 0;   // \sum{w\in V} { \frac{cf(w)}{|coll|} } WILL be same for all query
        for (Map.Entry<String, PerTermStat> entrySet : collStat.perTermStat.entrySet())
            sum_rcf_V += entrySet.getValue().rcf;

        //* sum_rcf_V contains the sum of relative collection frequency of all terms in the collection
        //* sum_rcf_V => SAME FOR A COLLECTION

        List<DocumentVector> initialRetrvDocVecs; //initially retrieved document vectors
        HashMap initialRetrvTerms;        // terms from initially retrieved documents

        for (TRECQuery query : queries) {
            TopScoreDocCollector collector = TopScoreDocCollector.create(numWanted);
            Query luceneQry;
            luceneQry = trecQueryparser.getAnalyzedQuery(query);

            //System.out.println(luceneQry);

            searcher.search(luceneQry, collector);
            topDocs = collector.topDocs();
            hits = topDocs.scoreDocs;
            if(hits == null)
                System.out.println("Nothing found");

//            System.out.println("Retrieved results for query: " + query.id);
            int hits_length = hits.length;
            //System.out.println("Retrieved Length: " + hits_length);

            StringBuffer resBuffer = new StringBuffer();
            for (int i = 0; i < hits_length; ++i) {
                int docId = hits[i].doc;
                Document d = searcher.doc(docId);
                resBuffer.append(query.qid).append("\tQ0\t").
                    append(d.get("docid")).append("\t").
                    append((i)).append("\t").
                    append(hits[i].score).append("\t").
                    append(runName).append("\n");                
            }
            resFileWriter.write(resBuffer.toString());

            // +++++ Clarity Score
            double sum_MLE_Q_R; // sum of MLE of terms of query Q in all retrieved documents (SAME FOR A SINGLE QUERY)

            double mle_Q_D;    // MLE of query Q in document d
            sum_MLE_Q_R = 0;

            // +++ initial retrieved vocabulary building and doc vector saving 

            initialRetrvDocVecs = new ArrayList<>();
            initialRetrvTerms = new HashMap<>();  // (term, rcf )

            //* for each retrieved document
            for (int i = 0; i < hits_length; i++) {
                int docId = hits[i].doc;
                //Document d = searcher.doc(docId);
                DocumentVector dv = new DocumentVector();
                dv = dv.getDocumentVector(docId, collStat);
                initialRetrvDocVecs.add(dv);
                PerTermStat pts;

                Iterator<Entry<String, PerTermStat>> iterator = dv.docVec.entrySet().iterator();
                //* all terms of the documents are putting into initialRetTerms hashmap
                while(iterator.hasNext()){
                    Map.Entry<String, PerTermStat> termEntry = iterator.next();
                    String term = termEntry.getKey();
                    pts = termEntry.getValue();
                    initialRetrvTerms.put(term, pts.rcf);
                    //System.out.println(term +" :: "+ pts.rcf);
                    // Note: Terms may be coming from multiple documents. Multiple
                    //      'put' operation in the hashmap will be resulting in only
                    //      re-entry of the same term in the hashmap, causes no effect.
                    //      Done in this way to avoid 'if' check of already existing terms.
                }

                //* product of MLE of each query term q (in Q) in document d calculation is performed here
                //* It is calculated here to utilize the 'for each doc' loop
                // for each of the query terms
//                System.out.println("Query: " +luceneQry.toString("content"));
                String title_terms[] = luceneQry.toString("content").split(" ");
                mle_Q_D = 1;
                for (String title_term : title_terms) {
                    pts = dv.docVec.get(title_term);
                    // pts == null if query-term is not present in the document

                    PerTermStat pts2 = collStat.perTermStat.get(title_term);
                    double rcf = (pts2==null)?0:pts2.rcf;
                    // rcf = relative collection frequency of the query term

                    mle_Q_D *= (lambda*((pts==null)?0:pts.rcf) + (1-lambda)*rcf);
                }
                // mle_Q_D => MLE of Q in a single document D
                sum_MLE_Q_R += mle_Q_D;
                // sum_MLE_Q_R => SAME FOR A SINGLE QUERY
            } //* for each retrieved document
            //System.out.println("All initial retrieved document vectors saved in initialRetDocVecs");
            //System.out.println("sum_MLE_Q_R: \\sum {MLE of Q in all docs of R } calculated");

            // --- initial retrieved vocabulary built and doc vector saved
            //* initialRetDocVec = all initially retrieved document vectors 
            //* initialRetTerms  = all terms in the initially retrieved documents

            /* +++ */
            double W_in_Rv = 0;         // all 'w' in Retrieved documents
            double W_in_V_minus_Rv = 0; // all 'w' in V - R

            double w_rcf = 0;           // relative 'collection' frequency of a term 'w' under consideration
            double sum_rcf_R = 0;           // sum of ( relative cf of all 'w' 

            // +++ For each term of the initially retrieved vocabulary
            Iterator<Entry<String, Double>> iterator = initialRetrvTerms.entrySet().iterator();
            while(iterator.hasNext()){

                Entry<String, Double> termEntry = iterator.next();  // a term entry from initialRetrievedVocabulary
                String w = termEntry.getKey();          // the collection term 'w' under consideration
                w_rcf = termEntry.getValue();           // rcf of term 'w' under consideration
                sum_rcf_R += w_rcf;                     // sum of relative frequnecies of retrieved terms
                double mle_w_d = 0;                 // 
                //System.out.println(term +" :: "+ pts.rcf);

                // For each document of the initially retrieved documents(500)
                for (int i = 0; i < hits_length; i++) {
                    DocumentVector dv = initialRetrvDocVecs.get(i);
                    PerTermStat pts = dv.docVec.get(w);

                    // pts == null means retrieved term 'w' is not in the document
                    mle_w_d += ( lambda*((pts==null)?0:pts.rcf) + (1-lambda)*w_rcf );

                }   // for each retrieved document
                W_in_Rv += (mle_w_d * Math.log(sum_MLE_Q_R * mle_w_d / w_rcf)/Math.log(2));
                //System.out.println(mle_w_d);
            }   // for each term of the retrieved vocabulary
            W_in_Rv *= sum_MLE_Q_R;
            // --- For each term of the initially retrieved vocabulary
            // Initialised:
            //      sum_rcf_R = sum of relative CF of retrieved terms
            /* --- */

            /* +++ */
            //for each terms in V - Rv { 
            W_in_V_minus_Rv = (1-lambda) * hits_length * (collStat.colSize-initialRetrvTerms.size())
                * sum_MLE_Q_R*Math.log(sum_MLE_Q_R * (1-lambda) * hits_length)/Math.log(2)
                * (sum_rcf_V - sum_rcf_R);
            /* --- */

//            System.out.println(W_in_Rv);
//            System.out.println(W_in_V_minus_Rv);
            System.out.print(query.qid+": ");
            System.out.println("C.S. :" + (W_in_Rv + W_in_V_minus_Rv) );
//            System.exit(0);
        } //* for each query
        resFileWriter.close();
    }

    public static void main(String[] args) throws IOException, Exception {
        if(0 == args.length) {
            System.out.println("Usage: java ClarityScore <propfile-path>");
        }
        args = new String[2];
        args[0] = "idf-trec8.properties";
        ClarityScore claritySocre = new ClarityScore(args[0]);

        claritySocre.retrieveAll();
    }

}
