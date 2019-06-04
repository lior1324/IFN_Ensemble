package IFN;

import DataSet.DataSet;
import DataSet.chiSquareUtil;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;

public class InfoFuzzyNetwork {
    public static chiSquareUtil chi_square = new chiSquareUtil();

    public int max_depth;
    public int max_feature;
    public int n_classes_;
    public int n_features_;
    public DataSet ds;
    public double significance;
    public boolean preprune;
    public ArrayList<IFNLayer> layerArr;
    public ArrayList<String> splitted_att;
    public boolean nominalPreprune;

    public InfoFuzzyNetwork(int max_depth, int max_feature, double significance, boolean preprune, boolean nominalPreprune){
        this.max_depth = max_depth;
        this.max_feature = max_feature;
        this.n_classes_ = 0;
        this.n_features_ = 0;
        this.ds = null;
        this.significance = significance;
        this.preprune = preprune;
        this.layerArr = new ArrayList<>();
        this.splitted_att = new ArrayList<>();
        this.nominalPreprune = nominalPreprune;
    }

    public void fit(DataSet ds, boolean print){
        this.ds = ds;
        this.n_features_ = this.ds.getColumnName().size()-1;
        int n_feature = this.n_features_;
        ArrayList<String> att_random = get_random_att();
        IFNNode startNode = new IFNNode(ds.n_samples,null,this,att_random);
        startNode.fit(this.ds);
        IFNLayer layer0 = new IFNLayer(0,null,this,att_random);
        layer0.NodesArr.add(startNode);
        layer0.fit();
        if(print){
            System.out.print(layer0.splitBy+"  ,  ");
            print_t(layer0.splitBy_t_arr);
        }
        this.layerArr.add(layer0);
        if(this.max_depth == -1){
            for(int i = 0; i < n_feature; i++){
                att_random = get_random_att();
                IFNLayer temp_layer = this.layerArr.get(this.layerArr.size()-1).buildNewLayer(att_random);
                if(temp_layer == null) break;
                this.layerArr.add(temp_layer);
                if(print){
                    System.out.print(temp_layer.splitBy+"  ,  ");
                    print_t(temp_layer.splitBy_t_arr);
                }
            }
        }
        else{
            for(int i = 0; i < max_depth; i++){
                att_random = get_random_att();
                IFNLayer temp_layer = this.layerArr.get(this.layerArr.size()-1).buildNewLayer(att_random);
                if(temp_layer == null) break;
                this.layerArr.add(temp_layer);
                if(print){
                    System.out.print(temp_layer.splitBy+"  ,  ");
                    print_t(temp_layer.splitBy_t_arr);
                }
            }
        }
    }

    public int[] predict(double[][] X){
        int[] preds = new int[X.length];
        for(int i = 0; i < X.length; i++){
            IFNNode predNode = this.layerArr.get(0).NodesArr.get(0);
            boolean flag = false;
            while((predNode.next_nodes.size() > 0) && (!(flag))){
                boolean flag1 = false;
                String feature_split_name = predNode.feature_split;
                if(feature_split_name.contains("nominal")){
                    for(int j = 0; j < predNode.next_nodes.size(); j++){
                        double str1 = Double.parseDouble(predNode.next_nodes.get(j).prev_node_feature_split_value);
                        double str2 = X[i][this.ds.getIndexByColumn(feature_split_name)];
                        if(str1 == str2){
                            predNode = predNode.next_nodes.get(j);
                            flag1 = true;
                            break;
                        }
                    }
                    if(!(flag1)) flag = true;
                }
                else{
                    for(int j = 0; j < predNode.next_nodes.size(); j++){
                        String[] min_max_arr = predNode.next_nodes.get(j).prev_node_feature_split_value.split(" - ");
                        double min_interval = Double.parseDouble(min_max_arr[0]);
                        double max_interval = Double.parseDouble(min_max_arr[1]);
                        double current = X[i][this.ds.getIndexByColumn(feature_split_name)];
                        DecimalFormat df = new DecimalFormat("#.####");
                        df.setRoundingMode(RoundingMode.HALF_UP);
                        current = Double.parseDouble(df.format(current));
                        if((current <= max_interval) && (current >= min_interval)){
                            predNode = predNode.next_nodes.get(j);
                            flag1 = true;
                            break;
                        }
                    }
                    if(!(flag1)) flag = true;
                }
            }
            preds[i] = get_key_max_val(predNode.targets_prob);
        }
        return preds;
    }

    private ArrayList<String> get_random_att(){
        ArrayList<String> res = new ArrayList<>();
        res.addAll(this.ds.getColumnName());
        res.remove(res.size()-1);
        if(this.max_feature == -1){
            return res;
        }
        for(int i = 0; i < res.size(); i++){
            if(this.splitted_att.contains(res.get(i))){
                res.remove(res.get(i));
            }
        }
        Collections.shuffle(res);
        ArrayList<String> resS = new ArrayList<>();
        if(this.max_feature < res.size()){
            for(int i = 0; i < this.max_feature; i++){
                resS.add(res.get(i));
            }
        }
        else{
            resS.addAll(res);
        }
        return resS;
    }

    private void print_t(ArrayList<Double> t_arr){
        if(t_arr != null){
            for(int i = 0; i < t_arr.size(); i++){
                System.out.print(t_arr.get(i)+" | ");
            }
        }
        System.out.println("");
    }

    private int get_key_max_val(Hashtable<Integer,Double> ht){
        double max = -1;
        int max_key=0;
        Set<Integer> keys = ht.keySet();
        Iterator it = keys.iterator();
        while(it.hasNext()) {
            int threshold = (int) it.next();
            double mi_from_dic = ht.get(threshold);
            if(mi_from_dic > max){
                max = mi_from_dic;
                max_key = threshold;
            }
        }
        return max_key;
    }

    public Double[][] predict_proba(double[][] X){
        Double[][] preds = new Double[X.length][this.ds.n_classes];
        for(int i = 0; i < X.length; i++) {
            IFNNode predNode = this.layerArr.get(0).NodesArr.get(0);
            boolean flag = false;
            while ((predNode.next_nodes.size() > 0) && (!(flag))) {
                boolean flag1 = false;
                String feature_split_name = predNode.feature_split;
                if (feature_split_name.contains("nominal")) {
                    for (int j = 0; j < predNode.next_nodes.size(); j++) {
                        double str1 = Double.parseDouble(predNode.next_nodes.get(j).prev_node_feature_split_value);
                        double str2 = X[i][this.ds.getIndexByColumn(feature_split_name)];
                        if (str1 == str2) {
                            predNode = predNode.next_nodes.get(j);
                            flag1 = true;
                            break;
                        }
                    }
                    if (!(flag1)) flag = true;
                } else {
                    for (int j = 0; j < predNode.next_nodes.size(); j++) {
                        String[] min_max_arr = predNode.next_nodes.get(j).prev_node_feature_split_value.split(" - ");
                        double min_interval = Double.parseDouble(min_max_arr[0]);
                        double max_interval = Double.parseDouble(min_max_arr[1]);
                        double current = X[i][this.ds.getIndexByColumn(feature_split_name)];
                        DecimalFormat df = new DecimalFormat("#.####");
                        df.setRoundingMode(RoundingMode.HALF_UP);
                        current = Double.parseDouble(df.format(current));
                        if ((current <= max_interval) && (current >= min_interval)) {
                            predNode = predNode.next_nodes.get(j);
                            flag1 = true;
                            break;
                        }
                    }
                    if (!(flag1)) flag = true;
                }
            }
            for(int j = 0; j < this.ds.n_classes; j++){
                if(predNode.targets_prob.containsKey(j)){
                    preds[i][j] = predNode.targets_prob.get(j);
                }
                else{
                    preds[i][j] = 0.0;
                }
            }
        }
        return preds;
    }

}
