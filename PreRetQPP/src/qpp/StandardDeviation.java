/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package qpp;


public class StandardDeviation {

    public static double standardDeviation(double a[]) {

        double sd = 0;
        double average = 0;

        for(int i=0; i<a.length; i++) {
            average += a[i];
        }

        average /= a.length;

        for(int i=0; i<a.length; i++) {
            sd += ((a[i] - average) * (a[i] - average)) / a.length;
        }

//        System.out.print(Math.sqrt(sd) - average+"\t");
        return Math.sqrt(sd);
    }
}
