
import matplotlib.pyplot as plt
import metrics
import troubleshooting
import random
import pandas as pd
from numpy import testing

d = troubleshooting.Diagnosis(code_name='Refrac3_local')
d.eval(testset_name='train_r100', task='triangle', epoch=300)

def triple_convergence(sel_word, show_plot=False):
    """Visualize and assert the mean activation in on and off nodes over time is equal across 1) troubleshooting.Diagnosis(), evaluate.TestSet(), and metrics.OutputofXXXTarget()
    """
    d.set_target_word(sel_word)

    # Trouble shooter module (That is new, containing live evaluation and testSet module)
    trouble_shooter_act = d.word_sem_df.loc[d.word_sem_df.variable == 'act'].groupby(['target_act', 'timetick']).mean().reset_index()
    trouble_shooter_act0 = trouble_shooter_act.loc[trouble_shooter_act.target_act==0., 'value'].to_list()
    trouble_shooter_act1 = trouble_shooter_act.loc[trouble_shooter_act.target_act==1., 'value'].to_list()

    # Evaluator module (Within Troubleshooter)
    evaluator_act = d.df.loc[(d.df.word==sel_word) & (d.df.output_name=='sem')][['act0', 'act1']].reset_index()
    evaluator_act0 = evaluator_act['act0'].to_list()
    evaluator_act1 = evaluator_act['act1'].to_list()

    # Metric module that Evaluator module calls 
    metric_act0 = metrics.OutputOfZeroTarget()
    metric_act1 = metrics.OutputOfOneTarget()

    word_idx = d.testset_package['item'].index(sel_word)
    target_sem_word = d.testset_package['sem'][word_idx,:]  # Dimension [output node]
    act_sem_word = d.y_pred['sem'][:,word_idx,:] # Dimension [timestep, output node]

    met_act1 = metric_act1.item_metric(target_sem_word, act_sem_word)
    met_act0 = metric_act0.item_metric(target_sem_word, act_sem_word)

    # Assertion at 1e-6
    testing.assert_allclose(evaluator_act0, trouble_shooter_act0, rtol=1e-6)
    testing.assert_allclose(trouble_shooter_act0, met_act0, rtol=1e-6)
    testing.assert_allclose(evaluator_act1, trouble_shooter_act1, rtol=1e-6)
    testing.assert_allclose(trouble_shooter_act1, met_act1, rtol=1e-6)
    
    if show_plot:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20,10))

        ax0.plot(trouble_shooter_act0, label='trouble_shooter act0')
        ax0.plot(trouble_shooter_act1, label='trouble_shooter act1')
        ax0.legend()

        ax1.plot(evaluator_act0, label='evaluator act0')
        ax1.plot(evaluator_act1, label='evaluator act1')
        ax1.legend()

        ax2.plot(met_act0, label='metrics act0')
        ax2.plot(met_act1, label='metrics act1')
        ax2.legend()

        fig.suptitle(f'In word "{sel_word}"')

    print(f'Activation is consistent in word: {sel_word}')

if __name__ == '__main__':
    word = random.sample(list(d.all_words.values()), 1)
    triple_convergence(word[0])