package msmarco;
import java.util.HashMap;
import java.util.Map;

public class TermDistribution {
    String queryTerm;
    Map<String, Double> cooccurProbs;

    TermDistribution(String queryTerm) {
        this.queryTerm = queryTerm;
        this.cooccurProbs = new HashMap<>();
    }

    void add(String term_and_wt_token) {
        String[] term_wt_token_parts = term_and_wt_token.split("\\" + SupervisedRLM.DELIM);
        String term = term_wt_token_parts[0];
        if (term_wt_token_parts.length<2)
            System.err.println("Problem for token|" + term_and_wt_token + "|");

        double wt = Double.parseDouble(term_wt_token_parts[1]);
        cooccurProbs.put(term, wt);
    }

    void update(String qterm, Map<String, Double> docTermWts) {
        double wt = 0;
        double cooccur = 0;
        Double p_q_d = docTermWts.get(qterm); // this can be null

        Double cooccur_accumulated = null;
        for (Map.Entry<String, Double> e: docTermWts.entrySet()) {
            String w = e.getKey();
            Double p_w_d = e.getValue(); // this can't be null coz this we get by iterating
            cooccur = p_q_d!=null? p_w_d * p_q_d : 0;

            cooccur_accumulated = cooccurProbs.get(w);
            if (cooccur_accumulated == null) {
                cooccur_accumulated = new Double(0.0);
            }
            if (cooccur > 0)
                cooccurProbs.put(w, cooccur_accumulated + cooccur);  // this is the weight of 'w' (and not q)
        }
    }

    double klDiv(Map<String, Double> termWts) {
        double kldiv = 0;
        for (Map.Entry<String, Double> e: cooccurProbs.entrySet()) {
            String w = e.getKey();
            double p_w_R = e.getValue();
            Double p_w_d = termWts.get(w);
            if (p_w_d == null) continue; // avoid 0 division error
            if (p_w_R == 0) continue;

            kldiv += p_w_R * Math.log(p_w_R/p_w_d);
        }
        return kldiv;
    }
}