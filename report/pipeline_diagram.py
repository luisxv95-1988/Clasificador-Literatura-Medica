from graphviz import Digraph
def main(out='report/pipeline_classifier.png'):
    dot = Digraph(comment='Medical Literature Classifier Pipeline', format='png')
    dot.attr(rankdir='LR')
    for k,v in {
        'A':'Input Data\n(title + abstract)',
        'B':'Preprocessing\n- Clean\n- Concatenate',
        'C':'Vectorization\nTF-IDF',
        'D':'Classifier\nLogistic Regression (OvR)',
        'E':'Predicted Labels\n(group_predicted)',
        'F':'Evaluation\n- Weighted/Micro/Macro F1\n- Confusion Matrix',
        'G':'Visualizations\n(V0 Bonus)'
    }.items():
        dot.node(k, v, shape='box')
    dot.edges(['AB','BC','CD','DE']); dot.edge('D','F'); dot.edge('E','F'); dot.edge('F','G')
    dot.render(out.replace('.png',''), format='png', cleanup=True)
if __name__ == '__main__':
    main()
