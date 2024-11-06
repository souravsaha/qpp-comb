/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;


class PerTermStat {

    String  term;
    long    cf;     // cf with respect to collection; tf with respect to document
    long    df;     // df with respect to collection; 1 with respect to document
    double  idf;    // inverse-document-frequency
    // idf = 1 + log(#-of-Document-in-collection / df)
    double  rcf;    // relative collection frequency in collection as a whole
    // rcf(w,Coll) = cf(w) / |collection-size|
    // rcf(w,Docu) = cf(w) / |document-size|

    public PerTermStat() {        
    }

    public PerTermStat(String term, long cf, long df) {
        this.term = term;
        this.cf = cf;
        this.df = df;
    }

    public PerTermStat(String term, long cf, long df, double idf, double rcf) {
        this.term = term;
        this.cf = cf;
        this.df = df;
        this.idf = idf;
        this.rcf = rcf;
    }
}
