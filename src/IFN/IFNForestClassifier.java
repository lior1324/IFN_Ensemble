package IFN;

import DataSet.DataSet;
import java.util.Random;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.concurrent.*;

public class IFNForestClassifier {
    public boolean bootstrap;
    public int max_depth;
    public int max_features;
    public int n_estimators;
    public int n_threads;
    public DataSet ds;
    public ArrayList<InfoFuzzyNetwork> clf_arr;
    public double significance;
    public boolean preprune;
    public double oob_score;
    public int y_values_num;
    public boolean nominalPreprune;

    public IFNForestClassifier(boolean bootstrap, int max_depth, int max_features, int n_estimators, int n_threads,double significance,boolean preprune, boolean nominalPreprune){
        this.bootstrap = bootstrap;
        this.max_depth = max_depth;
        this.max_features = max_features;
        this.n_estimators = n_estimators;
        this.n_threads = n_threads;
        this.clf_arr = new ArrayList<>();
        this.significance = significance;
        this.preprune = preprune;
        this.oob_score = 0;
        this.nominalPreprune = nominalPreprune;
    }

    public void fit(DataSet ds,int y_values_num){
        this.ds = ds;
        this.y_values_num = y_values_num;
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(this.n_threads);
        for(int i = 0; i < this.n_estimators; i++){
            int k = i;
            executor.execute(new Runnable() {
                @Override
                public void run() {
                    ifn_thread(k);
                }
            });
        }
        executor.shutdown();
        try{
            executor.awaitTermination(100,TimeUnit.HOURS);
        }
        catch (Exception e){

        }
        this.oob_score = (double) this.oob_score / this.ds.n_samples;
    }

    private Hashtable<String,ArrayList<Integer>> get_bootstrap(int n_samples){
        Random rand = new Random();
        Hashtable<String,ArrayList<Integer>> res = new Hashtable<>();
        res.put("bootstrap",new ArrayList<>());
        res.put("oob",new ArrayList<>());
        for(int i = 0; i < this.ds.n_samples; i++){
            res.get("bootstrap").add(rand.nextInt(this.ds.n_samples));
        }
        for(int i = 0; i <this.ds.n_samples; i++){
            if(!(res.get("bootstrap").contains(i))){
                res.get("oob").add(i);
            }
        }
        return res;
    }

    public int[] predict(double[][] X){
        int[] res = new int[X.length];
        ArrayList<Double[][]> clf_preds = new ArrayList<>();
        for(int i = 0; i<X.length; i++){// init the data struct with 0.0
            clf_preds.add(new Double[clf_arr.size()][this.ds.n_classes]);
        }
        /// fill the data struct from predict proba
        for(int i = 0; i < this.clf_arr.size(); i++){///i is the clf number
            Double[][] temp_pred = clf_arr.get(i).predict_proba(X);
            for(int j = 0; j < temp_pred.length; j++){///j is the sample number
                for(int k = 0; k < temp_pred[0].length; k++){// k is diff y values prob
                    clf_preds.get(j)[i][k] = temp_pred[j][k];
                }
            }
        }
        ///combine to int[] for int to each sample
        for(int i = 0; i < clf_preds.size(); i++){ //i is sample number
            double[] temp = new double[this.ds.n_classes];
            for(int j = 0; j < temp.length; j++){//j is diff y value
                for(int k = 0; k < clf_arr.size(); k++){//k is cld number
                    temp[j] += clf_preds.get(i)[k][j];
                }
            }
            res[i] = get_idx_max_value(temp);
        }
        return res;
    }
    private int get_idx_max_value(double[] arr){
        int idx = 0;
        for(int i = 0; i < arr.length; i++){
            if(arr[i] > arr[idx]){
                idx = i;
            }
        }
        return idx;
    }
    private int maxRepeating(int[] arr, int n, int k)
    {
        for (int i = 0; i< n; i++) arr[(arr[i]%k)] += k;
        int max = arr[0], result = 0;
        for (int i = 1; i < n; i++)
        {
            if (arr[i] > max)
            {
                max = arr[i];
                result = i;
            }
        }
        return result;
    }

    private void ifn_thread(int i){
        InfoFuzzyNetwork clf = new InfoFuzzyNetwork(this.max_depth,this.max_features,this.significance,this.preprune,this.nominalPreprune);
        Hashtable<String,ArrayList<Integer>> bootstrap = get_bootstrap(this.ds.n_samples);
        DataSet oob_dataset;
        if(this.bootstrap){
            oob_dataset = new DataSet(ds,bootstrap.get("oob"));
            DataSet boot = new DataSet(ds,bootstrap.get("bootstrap"));
            clf.fit(boot,false);
        }
        else{
            oob_dataset = ds;
            DataSet boot = ds;
            clf.fit(boot,false);
        }

        int[] preds = clf.predict(oob_dataset.X);
        double oob = 1 - DataSet.confusion_metrix(preds,oob_dataset.y,this.y_values_num,false);
        synchronized (this){
            this.oob_score = this.oob_score + oob;
            this.clf_arr.add(clf);
        }
    }
}
