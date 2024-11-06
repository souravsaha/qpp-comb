/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;

import static common.CommonVariables.FIELD_WORD_NAME;
import static common.CommonVariables.FIELD_WORD_VEC;
import common.InputOutput;
import common.TRECQuery;
import common.TRECQueryParser;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.xml.sax.SAXException;
import static qpp.NQC.returnNQC;
import wordvectors.WordVec;
import wordvectors.WordVecs;


public class FiniteGMMEstimation {

    Properties prop;
    WordVecs wordVecs;
    static IndexReader wvecReader;
    static IndexSearcher wvecSearcher;

    IndexReader     reader;
    IndexSearcher   searcher;
    String          indexPath;
    File            indexFile;
    Analyzer        analyzer;
    String          stopFilePath;
    String queryPath;
    File            queryFile;      // the query file
    int             queryFieldFlag; // 1. title; 2. +desc, 3. +narr
    String fieldToSearch;
    List<common.TRECQuery> queries;
    TRECQueryParser trecQueryparser;
    HashMap<String, List<TermSimilarity>> termNNHash;
    int no_of_componentes;

    String nn_output_file;
    String proba_input_file;
    String cmd;

    int colDocCount;    // collection document count
    String resPath;
    private String mapPath;
    int noOfTopDocs;
    double alpha;
    TopScoreDocCollector collector;
    /**
     * Word Vector hashmap.
     */
    static HashMap<String, WordVec> wvecMap = new HashMap();
    String pythonCodePath;

    public FiniteGMMEstimation(String propPath) throws FileNotFoundException, IOException, SAXException, Exception {

        prop = new Properties();
        prop.load(new FileReader(propPath));        
        String loadFrom = prop.getProperty("wvecs.indexPath");
//        File indexDir = new File(loadFrom);

        loadWordVecsInMem(propPath);

        // +++++ setting the analyzer with English Analyzer with Smart stopword list
        stopFilePath = prop.getProperty("stopFilePath");
        common.EnglishAnalyzerWithSmartStopword engAnalyzer = new common.EnglishAnalyzerWithSmartStopword(stopFilePath);
        analyzer = engAnalyzer.setAndGetEnglishAnalyzerWithSmartStopword();
        // ----- analyzer set: analyzer

        //+++++ index path setting 
        indexPath = prop.getProperty("indexPath");
        indexFile = new File(indexPath);

        Directory indexDir = FSDirectory.open(indexFile.toPath());

        if (!DirectoryReader.indexExists(indexDir)) {
            System.err.println("Index doesn't exists in "+indexPath);
            System.exit(1);
        }
        // +++
        termNNHash = new HashMap<>();
        String nnPath = prop.getProperty("wvecs.nnPath");
        readNNFile(nnPath, 30);
        // ---
        //----- index path set
        /* setting query path */
        queryPath = prop.getProperty("queryPath");
        queryFile = new File(queryPath);
        /* query path set */
        fieldToSearch = prop.getProperty("fieldToSearch");//"full-content";

        /* constructing the query */
        trecQueryparser = new common.TRECQueryParser(queryPath, analyzer, fieldToSearch);
        queries = constructQueries();
        /* constructed the query */

        /* setting reader and searcher */
        reader = DirectoryReader.open(FSDirectory.open(indexFile.toPath()));

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new CustomSimilarity());   // search of the documents containing the query terms.
//        searcher.setSimilarity(new BM25Similarity());   // search of the documents containing the query terms.

        resPath = prop.getProperty("resPath");
        mapPath = prop.getProperty("mapPath");
        // no of top docs to be used for NQC
        noOfTopDocs = Integer.parseInt(prop.getProperty("noOfTopDocs", "100"));
        alpha = Double.parseDouble(prop.getProperty("alpha", "0.6"));
//        System.out.println("Alpha set to "+alpha);
        nn_output_file = "nn-vectors.txt";
        proba_input_file = "proba.txt";

        // GMM parameters
        String pythonPath = prop.getProperty("pythonPath");
        pythonCodePath = pythonPath + " " + prop.getProperty("pythonCodePath") + " ";
        no_of_componentes = Integer.parseInt(prop.getProperty("no_of_componentes", "8"));
        cmd = pythonCodePath + nn_output_file + " " + proba_input_file + " " + no_of_componentes;
        System.out.println(cmd);
    }

    private List<common.TRECQuery> constructQueries() throws Exception {

        trecQueryparser.queryFileParse();
        return trecQueryparser.queries;
    }

    private void readNNFile(String nnPath, int nnCount) throws FileNotFoundException, IOException {

        FileInputStream fis = new FileInputStream(new File(nnPath));
        String rootTerm_NNWithSim[];        // root term, its NN terms along with similarity with root term
        List<TermSimilarity> nnList;
 
	//Construct BufferedReader from InputStreamReader
	BufferedReader br = new BufferedReader(new InputStreamReader(fis));
 
	String line = null;
        line = br.readLine();   // read first line containing <no.of.word> [SPACE] <no.of.NNs>
        int noOfWords = Integer.parseInt(line.split(" ")[0]);
        int noOfNNs = Integer.parseInt(line.split(" ")[1]);

        //System.out.println(noOfWords + " " + noOfNNs);
        
        while ((line = br.readLine()) != null) {
            rootTerm_NNWithSim = new String[noOfNNs];
            rootTerm_NNWithSim = line.split("\t");  // noOfNNs+1 fields; first: root-term; then noOfNNs no. of terms with similarity
            nnList = new ArrayList<>();
            String rootWord = rootTerm_NNWithSim[0];
            //System.out.println("Root word: " + rootWord + " " + rootTerm_NNWithSim.length);
            for (int i = 0; i < Math.min(noOfNNs, nnCount) ; i++) {
                String[] singleNNWithSim = rootTerm_NNWithSim[i+1].split(" ");
                nnList.add(new TermSimilarity(singleNNWithSim[0], Float.parseFloat(singleNNWithSim[1])));
            }
            termNNHash.put(rootWord, nnList);
	}
 
	br.close();

        System.out.println("NNs read into memory");
    }

    public void loadWordVecsInMem(String propFile) throws IOException {

        prop = new Properties();
        prop.load(new FileReader(propFile));        
        String loadFrom = prop.getProperty("wvecs.indexPath");
        File indexDir = new File(loadFrom);

        wvecReader = DirectoryReader.open(FSDirectory.open(indexDir.toPath()));
        wvecSearcher = new IndexSearcher(wvecReader);
        
        System.out.println("Loading wvecs in memory...");
        int numDocs = wvecReader.numDocs();
        for (int i=0; i<numDocs; i++) {
            Document d = wvecReader.document(i);
            String wordName = d.get(FIELD_WORD_NAME);
            String line = d.get(FIELD_WORD_VEC);
            WordVec wv = new WordVec(line);
            wv.normalize();
            wvecMap.put(wordName, wv);
        }

    }

    public WordVec getVecCached(String word) throws Exception {
        return wvecMap.get(word);
    }

    private double readFile(File fin) throws IOException {
	FileInputStream fis = new FileInputStream(fin);

	//Construct BufferedReader from InputStreamReader
	BufferedReader br = new BufferedReader(new InputStreamReader(fis));
 
	String line = null;

        line = br.readLine();
 
	br.close();
        
        return Double.parseDouble(line);
    }

    public List<Double> sumNormalize(List<Double> list) {

        List<Double> normalized = new ArrayList<>();

        double norm = 0;
        for (Double temp : list)
            norm += temp;

        for (Double temp : list)
            normalized.add(temp/norm);
        return normalized;
    }

    private double productAmbiguity(List<Double> proba_list) {
        double ambiguity = 1;
        for(Double d : proba_list)
            ambiguity *= d;
        return ambiguity;
    }

    private void processAll() throws Exception {
        
        String queryTerms[];

        HashMap<String, Double[]> queryScoreHash = InputOutput.readScoreFromResFile(this.resPath);
        HashMap<String, Double> queryAPHash = InputOutput.readAPFromFile(this.mapPath);

        double nqc, qpp;
        double allAP[], allQPP[];
        int processedQueryCount = 0;
        allAP = new double [queries.size()];
        allQPP = new double [queries.size()];

        // create runtime to execute external command
        Runtime rt;
        Process pr;

        for (TRECQuery query : queries) {
//            System.out.print(query.qid + " ");
            Double[] qScore = queryScoreHash.get(query.qid);
            Double queryAP = queryAPHash.get(query.qid);

            Query luceneQuery = trecQueryparser.getAnalyzedQuery(query);
            if(!luceneQuery.toString(fieldToSearch).isEmpty()) {
//                System.out.println(luceneQuery.toString(fieldToSearch));
                queryTerms = new String[luceneQuery.toString(fieldToSearch).split(" ").length];
                queryTerms = luceneQuery.toString(fieldToSearch).split(" ");
                int qtCount = 0; // query term count

                if (null != qScore && null != queryAP) {
                    double score[] = ArrayUtils.toPrimitive(qScore);

                    nqc = returnNQC(score, noOfTopDocs);
    //                System.out.println(query.qid+ " "+"\t" + returnNQC(score, noOfTopDocs));

                }
                else
                    nqc = 0;

                List<Double> proba_list = new ArrayList<>();
                for (String queryTerm : queryTerms) {
                    qtCount++;
//                    System.out.println(queryTerm);
                    
                    // get all the NNs for that particular query term 
                    List<TermSimilarity> nnTerms = new ArrayList<>();
                    nnTerms = termNNHash.get(queryTerm);

                    PrintWriter writer = new PrintWriter(nn_output_file, "UTF-8");
                    if(nnTerms != null) {
                        WordVec wv = wvecMap.get(queryTerm);
                        writer.println(wv.toString());
                        for (TermSimilarity t : nnTerms) {
                            wv = wvecMap.get(t.term);
                            if(wv != null) {
//                                System.out.println(t.term + " " + wv.toString());
                                writer.println(wv.toString());
                            }
                        }
                    }
                    writer.close();
                    rt = Runtime.getRuntime();

                    pr = rt.exec(cmd);

                    int exitVal = pr.waitFor();
                    if(pr.exitValue() != 0) {
                        System.out.println("Error executing the python code.");
                        System.out.println(pr.getErrorStream());
                    }
                    double proba = readFile(new File(proba_input_file));
                    proba_list.add(proba);
                } // for each query term

//                System.out.println(Arrays.toString(proba_list.toArray()));
//                proba_list = sumNormalize(proba_list);
//                System.out.println(Arrays.toString(proba_list.toArray()));
                
//                char ch = (char) System.in.read();
                qpp = productAmbiguity(proba_list);
                allAP[processedQueryCount] = queryAP;
//                alpha = 1;
                allQPP[processedQueryCount] = (alpha * qpp + (1-alpha) * nqc);
                processedQueryCount++;

            }
        } // for each query
        PearsonsCorrelation cor = new PearsonsCorrelation();
        KendallsCorrelation tau = new KendallsCorrelation();
        System.out.println("RH: "+cor.correlation(allAP, allQPP) + " TAU: "
            + tau.correlation(allAP, allQPP));

    }

    public static void main(String[] args) throws IOException, Exception {
//        FiniteGMMEstimation obj = new FiniteGMMEstimation("qpp-trec8.properties");
        FiniteGMMEstimation obj = new FiniteGMMEstimation("finite-mog.properties");

        int i;
        double j;
//        for (j=0.0; j<1.0; j+=0.1)
            for (i=5; i<=15; i++)
            {
//                obj.alpha = j;
                obj.no_of_componentes = i;
                obj.alpha = 0.1;
                System.out.println("Alpha set to "+obj.alpha+" ");
                System.out.println("desirableClusterSize set to "+obj.no_of_componentes);
                obj.processAll();
            }
    }
}
