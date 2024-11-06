/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wordvectors;

import static common.CommonVariables.FIELD_WORD_NAME;
import static common.CommonVariables.FIELD_WORD_VEC;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;


public class WordVecsIndexer {

    IndexWriter     wvecsWriter;
    Properties      prop;
    String          wvecsIndexPath;

    public WordVecsIndexer(String propFile) throws Exception {
        prop = new Properties();
        prop.load(new FileReader(propFile));                
        wvecsIndexPath = prop.getProperty("wvecs.index");        
    }

    public void writeIndex() throws Exception {
        
        IndexWriterConfig iwcfg = new IndexWriterConfig(
                new WhitespaceAnalyzer());
        iwcfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

        wvecsWriter = new IndexWriter(FSDirectory.open(new File(wvecsIndexPath).toPath()), iwcfg);
        
        String fileToRead = prop.getProperty("wvecs.txt");
        indexWordVectors(new File(fileToRead));
        wvecsWriter.close();
        
        storeClusterInfo();
    }
    
    Document constructDoc(String id, String line) throws Exception {        
        Document doc = new Document();
        doc.add(new Field(FIELD_WORD_NAME, id, Field.Store.YES, Field.Index.NOT_ANALYZED));        
        doc.add(new Field(FIELD_WORD_VEC, line,
                Field.Store.YES, Field.Index.NOT_ANALYZED, Field.TermVector.NO));
        return doc;        
    }

    void storeClusterInfo() throws Exception {
        int numClusters = Integer.parseInt(prop.getProperty("wvecs.numclusters", "5"));
        String clusterInfoBaseDir = prop.getProperty("wvecs.clusterids.basedir");
        String clusterInfoDirPath = clusterInfoBaseDir + "/" + numClusters;
        
        File clusterInfoFile = new File(clusterInfoDirPath);
        if (clusterInfoFile.isDirectory() && clusterInfoFile.exists()) {
            System.out.println("Cluster info already exists...");
            return;
        }

        // Create the directory...
        clusterInfoFile.mkdir();
        IndexWriterConfig iwcfg = new IndexWriterConfig(
                new WhitespaceAnalyzer());
        iwcfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

        IndexWriter clusterInfoFileWriter = new IndexWriter(FSDirectory.open(clusterInfoFile.toPath()), iwcfg);        
        clusterWordVecs(clusterInfoFileWriter, numClusters);
        clusterInfoFileWriter.close();        
    }

    /**
     * Cluster the words of the vocabulary using K-Means clustering.
     * @param clusterIndexWriter
     * @param numClusters
     * @throws Exception 
     */
    void clusterWordVecs(IndexWriter clusterIndexWriter, int numClusters) throws Exception {

        IndexReader wvecsReader = DirectoryReader.open(FSDirectory.open(new File(wvecsIndexPath).toPath()));
        int numWords = wvecsReader.maxDoc();
        KMeansPlusPlusClusterer<WordVec> wordClusterer = new KMeansPlusPlusClusterer<>(numClusters);
        List<WordVec> wordList = new ArrayList<>(numWords);

        // reading the word-vectors in wordList
        for (int i =0; i<numWords; i++) {
            Document doc = wvecsReader.document(i);
            WordVec wv = new WordVec(doc.get(FIELD_WORD_VEC));
            wordList.add(wv);
        }

        System.out.println("Clustering the entire vocabulary using K-Means. K = "+numClusters);
        List<CentroidCluster<WordVec>> wordClusters = wordClusterer.cluster(wordList);

        // save the cluster info. in index
        int clusterId = 0;
        for (CentroidCluster<WordVec> cluster : wordClusters) {
            List<WordVec> clusterPoints = cluster.getPoints();
            for (WordVec clusterPoint : clusterPoints) {
                Document clusterInfo = constructDoc(clusterPoint.getWord(), String.valueOf(clusterId));
                clusterIndexWriter.addDocument(clusterInfo);
            }
            clusterId++;
        }

        wvecsReader.close();

    }

    /**
     * Index the words of the entire vocabulary along with the associated vectors
     * @param file The .vec file.
     * @throws Exception 
     */
    void indexWordVectors(File file) throws Exception {

        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        
        final int batchSize = 10000;
        int count = 0;
        
        // Each line is word vector
        while ((line = br.readLine()) != null) {
            
            int firstSpaceIndex = line.indexOf(" ");
            String word = line.substring(0, firstSpaceIndex);
            Document luceneDoc = constructDoc(word, line);
            
            if (count%batchSize == 0) {
                System.out.println("Added " + count + " words...");
            }
            
            wvecsWriter.addDocument(luceneDoc);
            count++;
        }
        br.close();
        fr.close();
    }
    
    /* Use this to index the output of word2vec, i.e. instead
       of loading the word vectors from an in-memory hashmap
       keyed by a word, use Lucene search to retrieve the vector
       given a word. This makes it possible to run this on
       a limited memory environment. */
    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[1];
            System.out.println("Usage: java WordVecsIndexer <prop-file>");
            args[0] = "init.properties";
            System.out.println("Using properties file: "+args[0]);
        }
        else
            System.out.println("Using properties file: "+args[0]);

        try {
            WordVecsIndexer wvIndexer = new WordVecsIndexer(args[0]);
            wvIndexer.writeIndex();
            wvIndexer.storeClusterInfo();
        }
        catch (Exception ex) { ex.printStackTrace(); }
    }
    
}
