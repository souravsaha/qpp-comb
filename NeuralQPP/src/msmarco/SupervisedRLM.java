package msmarco;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.util.BytesRef;

import org.apache.lucene.store.FSDirectory;
import msmarco.AllRelRcds;
import msmarco.PerQueryRelDocs;

public class SupervisedRLM {

    IndexReader reader;
    IndexSearcher searcher;
    Similarity sim;
    AllRelRcds rels;
    Map<String, String> queries;
    Map<String, TermDistribution> termDistributions;

    static final String DELIM = "^";

    SupervisedRLM(String qrelFile, String queryFile) throws Exception {
        reader = DirectoryReader.open(FSDirectory.open(new File(Constants.MSMARCO_INDEX).toPath()));
        searcher = new IndexSearcher(reader);
        sim = new LMDirichletSimilarity(1000);
        searcher.setSimilarity(sim);
        rels = new AllRelRcds(qrelFile);
        queries = loadQueries(queryFile);
        termDistributions = new HashMap<>();
    }

    Map<String, Double> makeLMTermWts(int docId) throws Exception {
        BytesRef term;
        String termText;
        int tf;
        Map<String, Integer> vec = new HashMap<>();

        // In MSMARCO, Lucene doc offsets and ids are identical
        Terms tfvector = reader.getTermVector(docId, Constants.CONTENT_FIELD);
        TermsEnum termsEnum = tfvector.iterator(); // access the terms for this field

        while ((term = termsEnum.next()) != null) { // explore the terms for this field
            termText = term.utf8ToString();
            tf = (int)termsEnum.totalTermFreq();

            Integer wt = vec.get(termText);
            if (wt == null) {
                wt = 0;
            }
            wt++;
            vec.put(termText, wt);
        }

        double sumTf = vec.values().stream().mapToDouble(x->x).sum();
        return vec.entrySet()
            .stream()
            .collect(
                Collectors
                .toMap(
                    e->e.getKey(),
                    e-> {
                        try {
                            return Math.log(1+
                                Constants.LAMBDA/(1-Constants.LAMBDA)*
                                e.getValue()/(double)sumTf * // tf
                                reader.numDocs()/reader.docFreq(new Term(Constants.CONTENT_FIELD, e.getKey())) // idf
                            );
                        } catch (IOException ioException) {
                            ioException.printStackTrace();
                            return null;
                        }
                    }
                )
        );
    }

    Map<String, String> loadQueries(String queryFile) throws Exception {
        return
            FileUtils.readLines(new File(queryFile), StandardCharsets.UTF_8)
                .stream()
                .map(x -> x.split("\t"))
                .collect(Collectors.toMap(x -> x[0], x -> x[1])
            )
        ;
    }

    void fit(String qid, String qText) throws Exception {
        PerQueryRelDocs relDocIds = rels.getRelInfo(qid);
        if (relDocIds == null)
            return;

        String[] qTerms = qText.split("\\s+");
        for (String q: qTerms) {
            for (String relDocId: relDocIds.getRelDocs()) {
                Map<String, Double> termWts = makeLMTermWts(Integer.parseInt(relDocId));
                TermDistribution termDistribution = termDistributions.get(q);
                if (termDistribution == null) {
                    termDistribution = new TermDistribution(q);
                }
                termDistribution.update(q, termWts);
                termDistributions.put(q, termDistribution);
            }
        }
    }

    void fit() throws Exception {
        int count = 0;
        System.out.println("#Training queries: " + queries.entrySet().size());

        for (Map.Entry<String, String> e: queries.entrySet()) {
            fit(e.getKey(), e.getValue());
            if (count++ % 100 == 0)
                System.out.print(String.format("Trained for %d queries\r", count-1));
        }
        System.out.println();
    }

    void saveToDisk() {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(Constants.SAVED_MODEL))) {
            StringBuilder b = new StringBuilder();
            for (TermDistribution termDistribution: termDistributions.values()) {
                b.setLength(0);
                b.append(termDistribution.queryTerm).append("\t");
                for (Map.Entry<String, Double> e: termDistribution.cooccurProbs.entrySet()) {
                    b.append(e.getKey()).append(DELIM).append(e.getValue()).append(" ");
                }
                bw.write(b.toString());
                bw.newLine();
            }
        }
        catch (Exception ex) { ex.printStackTrace(); }
    }

    void loadFromDisk() {
        System.out.println("Loading trained Supervised RLM from disk...");
        try (BufferedReader br = new BufferedReader(new FileReader(Constants.SAVED_MODEL))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\t");
                if (parts.length < 2)
                    continue;

                String queryTerm = parts[0];
                TermDistribution termDistribution = new TermDistribution(queryTerm);
                String[] tokens = parts[1].split("\\s+");
                for (String token: tokens) {
                    termDistribution.add(token);
                }
                termDistributions.put(queryTerm, termDistribution);
            }
        }
        catch (Exception ex) { ex.printStackTrace(); }
    }

    Query makeQuery(String queryText) throws Exception {
        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        String[] tokens = MsMarcoIndexer.analyze(MsMarcoIndexer.constructAnalyzer(), queryText).split("\\s+");
        for (String token: tokens) {
            TermQuery tq = new TermQuery(new Term(Constants.CONTENT_FIELD, token));
            qb.add(new BooleanClause(tq, BooleanClause.Occur.SHOULD));
        }
        return (Query)qb.build();
    }

    // Use the supervised RLM to rerank results
    TopDocs rerank(String qid, String queryText, TopDocs retrievedRes) throws Exception {
        String[] queryTerms = queryText.split("\\s+");
        ScoreDoc[] rerankedScoreDocs = new ScoreDoc[retrievedRes.scoreDocs.length];

        int i = 0;
        double kldiv = 0, kldiv_q;

        for (ScoreDoc sd: retrievedRes.scoreDocs) {
            rerankedScoreDocs[i] = new ScoreDoc(sd.doc, sd.score);
            kldiv = 0;

            for (String qTerm: queryTerms) {
                TermDistribution termDistribution_q = this.termDistributions.get(qTerm);
                if (termDistribution_q != null) {
                    Map<String, Double> doc_term_wts = makeLMTermWts(sd.doc);
                    kldiv_q = termDistribution_q.klDiv(doc_term_wts);
                    if (Double.isNaN(kldiv_q)) {
                        System.err.println("NAN");
                        continue;
                    }
                    kldiv += kldiv_q;
                }
            }
            rerankedScoreDocs[i].score = (float)kldiv;
            i++;
        }

        rerankedScoreDocs = Arrays.stream(rerankedScoreDocs)
            .sorted((o1, o2) -> o1.score < o2.score? -1: o1.score==o2.score? 0 : 1) // sort ascending by distance
            .collect(Collectors.toList())
            .toArray(rerankedScoreDocs)
        ;

        float maxDist = rerankedScoreDocs[rerankedScoreDocs.length-1].score; // last one is the max dist
        for (ScoreDoc sd: rerankedScoreDocs) {
            sd.score = maxDist - sd.score; // convert kldiv distances to sims
        }

        return new TopDocs(rerankedScoreDocs.length, rerankedScoreDocs, rerankedScoreDocs[0].score);
    }

    public void retrieve(boolean rerank) throws Exception {
        Map<String, String> testQueries = loadQueries(Constants.QUERY_FILE_TRAIN);

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(Constants.RES_FILE))) {

            for (Map.Entry<String, String> e : testQueries.entrySet()) {

                String qid = e.getKey();
                String queryText = e.getValue();

                //queryText = MsMarcoIndexer.normalizeNumbers(queryText);
                Query luceneQuery = makeQuery(queryText);
                System.out.println(String.format("Retrieving for query %s: %s", qid, luceneQuery));

                TopDocs topDocs = searcher.search(luceneQuery, Constants.NUM_WANTED); // descending BM25
                topDocs = rerank? rerank(qid, queryText, topDocs): topDocs; // sort by ascending KLdiv

                AtomicInteger rank = new AtomicInteger(1);
                
                ScoreDoc[] hits;
                // for writing the top list docs
                StringBuffer resBuffer;
                FileWriter resFileWriter;
                hits = topDocs.scoreDocs;
                if(hits == null)
                    System.out.println("Nothing found");
                int hitLength = hits.length;
                resFileWriter = new FileWriter(Constants.RES_FILE, true);
                /* res file in TREC format with doc text (7 columns) */
                resBuffer = new StringBuffer();
                for (int i = 0; i < hitLength; ++i) {
                    int docId = hits[i].doc;
                    Document d = searcher.doc(docId);
                    resBuffer.append(qid).append("\tQ0\t").
                    append(d.get("id")).append("\t").
                    append((i+1)).append("\t").
                    append(hits[i].score).append("\t").
                    append("lmdir").append("\n");
    //                append(d.get(FIELD_BOW)).append("\n");
                }
                resFileWriter.write(resBuffer.toString());
                resFileWriter.close();
//                 for writing the top list docs                              
                
//                for (ScoreDoc sd : topDocs.scoreDocs) {
//                    bw.write(String.format(
//                            "%s\tQ0\t%d\t%d\t%.6f\tsample",
//                            qid, sd.doc, rank.getAndIncrement(), sd.score,
//                            //"sample_run"
//                            reader.document(sd.doc).get(Constants.CONTENT_FIELD)
//                    ));
//                    bw.newLine();
//                }
            }
        }
    }

    public static void main(String[] args) {
        final boolean TRAIN = false;
        final boolean RERANK = false;

        try {
            SupervisedRLM supervisedRLM = new SupervisedRLM(
                    Constants.QRELS_TRAIN,
                    Constants.QUERY_FILE_TRAIN
            );
            if (TRAIN) {
                // Train on qrels
                System.out.println("Training SRLM on the MSMARCO training set");
                supervisedRLM.fit();
                supervisedRLM.saveToDisk();
            }
            else {
                if (RERANK)
                    supervisedRLM.loadFromDisk();
            }

            // Retrieve on test
            System.out.println("Writing out results file in " + Constants.RES_FILE);
            supervisedRLM.retrieve(RERANK);
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}