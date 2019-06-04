package IFN;

import DataSet.DataSet;
import java.util.*;

public class IFNNode {
    public DataSet ds;
    public int n_samples;
    public ArrayList<IFNNode> next_nodes;
    public InfoFuzzyNetwork network;
    public Hashtable<Integer,Double> targets_prob;
    public Hashtable<String,Double> input_att_mi;
    public String feature_split;
    public String prev_node_feature_split_value;
    public Hashtable<String,Hashtable<Double,Boolean>> att_t_split;
    public Hashtable<Double,Double> right_interval;
    public Hashtable<Double,Double> left_interval;
    public ArrayList<String> random_feature_arr;
    public Hashtable<String,Hashtable<Double,Double>> input_att_mi_t;

    public IFNNode(int n_samples,String prev_node_feature_split_value,InfoFuzzyNetwork network,ArrayList<String> random_feature_arr){
        this.n_samples = n_samples;
        this.prev_node_feature_split_value = prev_node_feature_split_value;
        this.network = network;
        this.random_feature_arr = random_feature_arr;
        this.ds = null;
        this.next_nodes = new ArrayList<>();
        this.targets_prob = new Hashtable<Integer, Double>();
        this.input_att_mi = new Hashtable<String,Double>();
        this.input_att_mi_t = new Hashtable<String, Hashtable<Double, Double>>();
        this.feature_split = "";
        this.att_t_split = new Hashtable<>();
        this.right_interval = new Hashtable<Double, Double>();
        this.left_interval = new Hashtable<Double, Double>();
    }

    private int my_count(double x_v, int y_v, double[] self_x_iloc){
        int count = 0;
        for(int i = 0; i < self_x_iloc.length; i++){
            if((self_x_iloc[i] == x_v) && (this.ds.y[i] == y_v)){
                count++;
            }
        }
        return count;
    }

    private int[] my_count_special(double x_v, int y_v, int j, double[][] X_temp, int[] y_temp){
        int[] res = new int[2];
        for(int i = 0; i < y_temp.length; i++){
            if((X_temp[i][j] <= x_v) && (y_temp[i] == y_v)){
                res[0] = res[0] + 1;
            }
            else if((X_temp[i][j] > x_v) && (y_temp[i] == y_v)){
                res[1] = res[1] + 1;
            }
        }
        return res;
    }

    private Hashtable<Double,Double> calc_continuous_mi(int att_idx, DataSet temp){
        double[] x_temp_att_idx = temp.getIlocX(att_idx);
        ArrayList<Double> diff_target_values = temp.getColumnUniqueAtIndex(ds.getColumnName().size()-1);
        int x_size = temp.n_samples;
        double min1 = temp.minAtColumn(ds.getColumnByIndex(att_idx));
        double max1 = temp.maxAtColumn(ds.getColumnByIndex(att_idx));
        Hashtable<Double,Double> t_to_mi_dic = new Hashtable<>();
        ArrayList<Double> unique_net_x = this.network.ds.getColumnUniqueAtIndex(att_idx);
        for (int i = 0; i < unique_net_x.size(); i++) {
            double t = unique_net_x.get(i);
            if((min1 <= t) && (t <= max1)){
                t_to_mi_dic.put(t,calc_mi(att_idx,t,temp,diff_target_values,x_size,x_temp_att_idx));
            }
        }
        t_to_mi_dic.remove(temp.maxAtColumn(ds.getColumnByIndex(att_idx)));
        int[] self_y_iloc_0 = this.ds.y;
        int self_x_size = this.ds.n_samples;
        for(int i = 0; i < diff_target_values.size(); i++) {
            this.targets_prob.put(diff_target_values.get(i).intValue(),(this.ds.getYCount(diff_target_values.get(i).intValue())/(double)self_x_size));
        }
        return t_to_mi_dic;
    }

    private boolean significance_helper(double mi){
        return mi>0;
    }

    private double calc_mi(int att_idx,double threshold,DataSet temp,ArrayList<Double> diff_target_values,int x_size,double[] x_in_att_idx){
        double mi=0.0;
        double cond_x_less=0;
        double cond_x_more=0;
        for (int i = 0; i <x_in_att_idx.length ; i++) {
            if(x_in_att_idx[i]<=threshold){
                cond_x_less++;
            }
            else{
                cond_x_more++;
            }
        }
        cond_x_less= cond_x_less/x_size;
        cond_x_more=cond_x_more/x_size;
        int df_size=this.n_samples;
        for (int i = 0; i < diff_target_values.size(); i++) {
            int[] count_x_y = my_count_special(threshold,diff_target_values.get(i).intValue(),att_idx,temp.X,temp.y);
            int count_x_y_less = count_x_y[0];
            int count_x_y_more = count_x_y[1];
            double joint_less = (double) count_x_y_less/df_size;
            double joint_more = (double) count_x_y_more/df_size;
            double cond_less = (double) count_x_y_less/x_size;
            double cond_more = (double) count_x_y_more/x_size;
            double cond_y = (double)temp.getYCount(diff_target_values.get(i).intValue())/x_size;
            if(count_x_y_more!=0){
                mi += joint_more*IFNNode.math_log_2(cond_more / (cond_x_more * cond_y));
            }
            if(count_x_y_less!=0){
                mi += joint_less*IFNNode.math_log_2(cond_less / (cond_x_less * cond_y));
            }
        }
        return mi;
    }

    private static double math_log_2(double num){
        return Math.log(num) / Math.log(2);
    }

    private double nominal_mi(int att_idx){
        int x_size = this.ds.n_samples;
        double mi = 0;
        ArrayList<Double> X_loop = this.ds.getColumnUniqueAtIndex(att_idx);
        for(int i = 0; i < X_loop.size(); i++) {
            double j = X_loop.get(i);
            double[] self_x_iloc = this.ds.getIlocX(att_idx);
            int count = 0;
            for(int k = 0; k < self_x_iloc.length; k++){
                if(self_x_iloc[k] == j) count++;
            }
            double cond_x = (double)count / x_size;
            ArrayList<Double> y_loop = this.ds.getColumnUniqueAtIndex(ds.getColumnName().size()-1);
            for(int k1 = 0; k1 < y_loop.size(); k1++){
                int k = y_loop.get(k1).intValue();
                double count_x_y = my_count(j,k,self_x_iloc);
                double joint = count_x_y/this.n_samples;
                double cond = count_x_y/x_size;
                double cond_y = (double)this.ds.getYCount(k)/x_size;
                this.targets_prob.put(k,cond_y);
                if(count_x_y!=0) mi+=joint*IFNNode.math_log_2(cond/(cond_x*cond_y));
            }
        }
        return mi;
    }

    public void fit(DataSet ds){
        this.ds = ds;
        ArrayList<Double> y_unique = this.ds.getColumnUniqueAtIndex(this.ds.getColumnName().size()-1);
        double net_sig = this.network.significance;
        for(int i = 0; i < y_unique.size(); i++){
            this.targets_prob.put(y_unique.get(i).intValue(),0.0);
        }
        for(int i = 0; i < this.ds.getColumnName().size()-1; i++){ //move on all attributes
            String att_name_i = this.ds.getColumnByIndex(i);
            int att_idx = this.ds.getIndexByColumn(att_name_i);
            if((!(this.network.splitted_att.contains(att_name_i))) && (this.random_feature_arr.contains(att_name_i))){
                double mi = 0.0;
                if(att_name_i.contains("nominal")) mi = nominal_mi(att_idx);
                else{
                    this.input_att_mi_t.put(att_name_i,calc_continuous_mi(att_idx,this.ds));
                }

                if(/*this.network.preprune*/true){
                    if(att_name_i.contains("nominal")){
                        if(this.network.nominalPreprune){
                            double G = mi * this.n_samples * 2 * Math.log(2);
                            int freedom_degree = (this.ds.getColumnUniqueAtIndex(att_idx).size() - 1) * (y_unique.size()-1);
                            double chi_from_table = 0;
                            if(this.network.preprune){
                                chi_from_table = InfoFuzzyNetwork.chi_square.chi_Stat(freedom_degree,net_sig);
                            }
                            if(G > chi_from_table) this.input_att_mi.put(att_name_i,mi);
                            else this.input_att_mi.put(att_name_i,0.0);
                        }
                        else{
                            this.input_att_mi.put(att_name_i,mi);
                        }
                    }
                    else{
                        int freedom_degree = y_unique.size()-1;
                        double chi_from_table = 0;
                        if(this.network.preprune){
                            chi_from_table = InfoFuzzyNetwork.chi_square.chi_Stat(freedom_degree,net_sig);
                        }
                        Hashtable<Double,Boolean> DicToInset = new Hashtable<>();
                        ArrayList<Double> x_loop = this.network.ds.getColumnUniqueAtIndex(att_idx);
                        for(int j = 0; j < x_loop.size(); j++){
                            DicToInset.put(x_loop.get(j),false);
                        }
                        this.att_t_split.put(att_name_i,DicToInset);
                        double min1 = this.ds.minAtColumn(att_name_i);
                        double max1 = this.ds.maxAtColumn(att_name_i);
                        Set<Double> keys = this.input_att_mi_t.get(att_name_i).keySet();
                        Iterator it = keys.iterator();
                        while(it.hasNext()){
                            double threshold = (double)it.next();
                            double mi_from_dic = this.input_att_mi_t.get(att_name_i).get(threshold);
                            double G = calc_g_stat_continuous(this.ds, threshold, att_name_i, min1, max1, this.ds.y,y_unique);
                            if(G > chi_from_table){
                                this.input_att_mi_t.get(att_name_i).put(threshold,mi_from_dic);
                                this.att_t_split.get(att_name_i).put(threshold,true);
                            }
                            else{
                                this.input_att_mi_t.get(att_name_i).put(threshold,0.0);
                                this.att_t_split.get(att_name_i).put(threshold,false);
                            }
                        }
                    }
                }
                else{
                    if(att_name_i.contains("nominal")) this.input_att_mi.put(att_name_i,mi);
                    else{
                        this.input_att_mi_t.put(att_name_i,new Hashtable<>());
                        Hashtable<Double,Boolean> DicToInset = new Hashtable<>();
                        ArrayList<Double> x_loop = this.network.ds.getColumnUniqueAtIndex(att_idx);
                        for(int j = 0; j < x_loop.size(); j++){
                            DicToInset.put(x_loop.get(j),false);
                        }
                        this.att_t_split.put(att_name_i,DicToInset);
                        Set<Double> keys = this.input_att_mi_t.get(att_name_i).keySet();
                        Iterator it = keys.iterator();
                        while(it.hasNext()){
                            double threshold = (double)it.next();
                            double mi_from_dic = this.input_att_mi_t.get(att_name_i).get(threshold);
                            if(mi_from_dic > 0){
                                this.input_att_mi_t.get(att_name_i).put(threshold,mi_from_dic);
                                this.att_t_split.get(att_name_i).put(threshold,true);
                            }
                            else{
                                this.input_att_mi_t.get(att_name_i).put(threshold,0.0);
                                this.att_t_split.get(att_name_i).put(threshold,false);
                            }
                        }
                    }
                }
            }
            else{
                this.input_att_mi.put(att_name_i,0.0);
            }
        }
    }

    public void split(String att_splitted, ArrayList<Double> threshold_arr, ArrayList<String> att_random){
        this.feature_split = att_splitted;
        int i = this.ds.getIndexByColumn(att_splitted);
        double[] X_iloc = this.ds.getIlocX(i);
        double[][] X = this.ds.X;
        int[] y = this.ds.y;
        if(att_splitted.contains("nominal")){
            ArrayList<Double> x_loop = this.ds.getColumnUniqueAtIndex(i);
            for(int j1 = 0; j1 < x_loop.size(); j1++){
                double j = x_loop.get(j1);
                ArrayList<Integer> indexes = new ArrayList<>();
                for(int k = 0; k < X_iloc.length; k++){
                    if(X_iloc[k] == j) indexes.add(k);
                }
                DataSet slice = new DataSet(this.ds,indexes);
                IFNNode temp_node = new IFNNode(this.n_samples,j+"",this.network,att_random);
                temp_node.fit(slice);
                this.next_nodes.add(temp_node);
            }
        }
        else{
            int count = 0;
            ArrayList<Double> unique_att_values = this.network.ds.getColumnUniqueAtIndex(i);
            Collections.sort(unique_att_values);
            double min_t = Collections.min(unique_att_values);
            double max_t = threshold_arr.get(count);
            ArrayList<Integer> indexes = new ArrayList<>();
            for(int k = 0; k < X_iloc.length; k++){
                if((min_t <= X_iloc[k]) && (X_iloc[k] <= max_t)) indexes.add(k);
            }
            if(indexes.size() > 0){
                DataSet slice = new DataSet(this.ds,indexes);
                int idx = unique_att_values.indexOf(max_t);
                IFNNode temp_node = new IFNNode(this.n_samples, min_t+" - "+(unique_att_values.get(idx+1)-0.000001),this.network,att_random);
                temp_node.fit(slice);
                this.next_nodes.add(temp_node);
            }
            min_t = max_t;
            count++;
            int len_arr = threshold_arr.size();
            if(count != len_arr) max_t = threshold_arr.get(count);
            for(int v = 0; v < len_arr-1; v++){
                indexes = new ArrayList<>();
                for(int k = 0; k < X_iloc.length; k++){
                    if((min_t < X_iloc[k]) && (X_iloc[k] <= max_t)) indexes.add(k);
                }
                if(indexes.size() > 0){
                    DataSet slice = new DataSet(this.ds,indexes);
                    int idx_min = unique_att_values.indexOf(min_t);
                    int idx_max = unique_att_values.indexOf(max_t);
                    IFNNode temp_node = new IFNNode(this.n_samples, unique_att_values.get(idx_min+1)+" - "+(unique_att_values.get(idx_max+1)-0.000001),this.network,att_random);
                    temp_node.fit(slice);
                    this.next_nodes.add(temp_node);
                }
                min_t = max_t;
                count++;
                if(count != len_arr) max_t = threshold_arr.get(count);
            }
            min_t = threshold_arr.get(len_arr-1);
            max_t = Collections.max(unique_att_values);
            indexes = new ArrayList<>();
            for(int k = 0; k < X_iloc.length; k++){
                if((min_t < X_iloc[k]) && (X_iloc[k] <= max_t)) indexes.add(k);
            }
            if(indexes.size() > 0){
                DataSet slice = new DataSet(this.ds,indexes);
                int idx_min = unique_att_values.indexOf(min_t);
                IFNNode temp_node = new IFNNode(this.n_samples, unique_att_values.get(idx_min+1)+" - "+max_t,this.network,att_random);
                temp_node.fit(slice);
                this.next_nodes.add(temp_node);
            }
        }
    }

    public void global_discretization_in_node(double threshold, String att, ArrayList<Double> threshold_arr, boolean flag){
        boolean myflag = false;
        ArrayList<Double> temp_t_arr = threshold_arr;
        if(flag) temp_t_arr.add(threshold);
        Collections.sort(temp_t_arr);
        int t_idx = temp_t_arr.indexOf(threshold);
        ArrayList<Double> net_x_att = this.network.ds.getColumnUniqueAtIndex(this.ds.getIndexByColumn(att));
        double min_interval = 0.0;
        double max_interval = 0.0;
        if(t_idx == 0){
            min_interval = Collections.min(net_x_att);
            myflag = true;
        }
        else min_interval = temp_t_arr.get(t_idx - 1);
        if(t_idx == temp_t_arr.size()-1) max_interval = Collections.max(net_x_att);
        else max_interval = temp_t_arr.get(t_idx + 1);
        int i = this.ds.getIndexByColumn(att);
        DataSet freedom = get_slices(min_interval,max_interval,i,myflag);
        int freedom_degree = freedom.getColumnUniqueAtIndex(freedom.getColumnName().size()-1).size()-1;
        double chi_from_table = 0;
        if(this.network.preprune){
            chi_from_table = InfoFuzzyNetwork.chi_square.chi_Stat(freedom_degree,this.network.significance);
        }
        DataSet slice = get_slices(min_interval,threshold,i,myflag);
        int[] slice_y_iloc = slice.y;
        ArrayList<Double> slice_y_unique = slice.getColumnUniqueAtIndex(freedom.getColumnName().size()-1);
        if(slice.n_samples == 0){
            this.left_interval = new Hashtable<>();
        }
        else{
            double min1 = slice.minAtColumn(att);
            double max1 = slice.maxAtColumn(att);
            this.left_interval = calc_continuous_mi(i,slice);
            if(/*this.network.preprune*/true){
                Set<Double> keys = this.left_interval.keySet();
                Iterator it = keys.iterator();
                Hashtable<Double,Double> temp_dic_left = new Hashtable<>();
                while(it.hasNext()){
                    double threshold_unique_val = (double) it.next();
                    double val = this.left_interval.get(threshold_unique_val);
                    temp_dic_left.put(threshold_unique_val,chi_stat(val, chi_from_table, threshold_unique_val, att, slice, min1, max1, slice_y_iloc, slice_y_unique));
                }
                this.left_interval = temp_dic_left;
            }
        }
        slice = get_slices(threshold,max_interval,i,false);
        slice_y_iloc = slice.y;
        slice_y_unique = slice.getColumnUniqueAtIndex(freedom.getColumnName().size()-1);
        if(slice.n_samples == 0){
            this.right_interval = new Hashtable<>();
        }
        else{
            double min1 = slice.minAtColumn(att);
            double max1 = slice.maxAtColumn(att);
            this.right_interval = calc_continuous_mi(i,slice);
            if(/*this.network.preprune*/true){
                Set<Double> keys = this.right_interval.keySet();
                Iterator it = keys.iterator();
                Hashtable<Double,Double> temp_dic_right = new Hashtable<>();
                while(it.hasNext()){
                    double threshold_unique_val = (double) it.next();
                    double val = this.right_interval.get(threshold_unique_val);
                    temp_dic_right.put(threshold_unique_val,chi_stat(val, chi_from_table, threshold_unique_val, att, slice, min1, max1, slice_y_iloc, slice_y_unique));
                }
                this.right_interval = temp_dic_right;
            }
        }
    }

    private DataSet get_slices(double min_interval, double max_interval, int i, boolean myflag){
        double[] self_x_iloc = this.ds.getIlocX(i);
        if(myflag){
            ArrayList<Integer> indexes = new ArrayList<>();
            for(int k = 0; k < self_x_iloc.length; k++){
                if((min_interval <= self_x_iloc[k]) && (self_x_iloc[k] <= max_interval)) indexes.add(k);
            }
            return new DataSet(this.ds,indexes);
        }
        else{
            ArrayList<Integer> indexes = new ArrayList<>();
            for(int k = 0; k < self_x_iloc.length; k++){
                if((min_interval < self_x_iloc[k]) && (self_x_iloc[k] <= max_interval)) indexes.add(k);
            }
            return new DataSet(this.ds,indexes);
        }
    }

    private double chi_stat(double mi, double chi_from_table, double threshold, String att, DataSet ds, double min1, double max1, int[] slice_y_iloc, ArrayList<Double> slice_y_unique){
        double G = calc_g_stat_continuous(ds,threshold,att,min1,max1,slice_y_iloc,slice_y_unique);
        if(G > chi_from_table){
            this.att_t_split.get(att).put(threshold,true);
            return mi;
        }
        else{
            this.att_t_split.get(att).put(threshold,false);
            return 0.0;
        }
    }

    public ArrayList<Double> reduce_threshold(ArrayList<Double> t_arr, String att){
        ArrayList<Double> res = new ArrayList<>();
        for(int i = 0; i < t_arr.size(); i++){
            if(this.att_t_split.get(att).get(t_arr.get(i))){
                res = t_arr;
                Collections.sort(res);
                return res;
            }
        }
        return res;
    }

    private double calc_g_stat_continuous(DataSet ds, double t, String att, double min1, double max1, int[] slice_y_iloc, ArrayList<Double> slice_y_unique){
        double sum = 0;
        int i = this.ds.getIndexByColumn(att);
        DataSet slice_less_equal_t = get_slices(min1,t,i,true);
        DataSet slice_more_t = get_slices(t,max1,i,false);
        int[] slice_less_equal_t_iloc = slice_less_equal_t.y;
        int[] slice_more_t_iloc = slice_more_t.y;
        for(int j = 0; j < slice_y_unique.size(); j++){
            double target_value = slice_y_unique.get(j);
            int num = slice_less_equal_t.getYCount((int)target_value);
            double num_father = (double)ds.getYCount((int)target_value)/ds.n_samples;
            if(num > 0){
                sum += num*Math.log(num/(num_father*slice_less_equal_t.n_samples));
            }
            num = slice_more_t.getYCount((int)target_value);
            if(num > 0){
                sum += num*Math.log(num/(num_father*slice_more_t.n_samples));
            }
        }
        return 2*sum;
    }
}
