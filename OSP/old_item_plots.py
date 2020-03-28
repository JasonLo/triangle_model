# Old item level plot

sel = alt.selection(type='single', on='click', fields=['word'], empty='all')

strain_chart_items = alt.Chart(
    result_strain_items[lambda df: df['epoch'] == nEpo]).add_selection(
        sel).mark_line(point=True).encode(
            y='sse',
            x='unit_time',
            color='word',
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            column='frequency',
            row='pho_consistency',
            tooltip=['epoch', 'word', 'pho', 'output', 'acc', 'sse'])

strain_chart_items


def plot_development_item(self, plot_time_step=None):
    import altair as alt
    if plot_time_step is None: plot_time_step = self.cfg.n_timesteps - 1

    sel = alt.selection(type='single',
                        on='click',
                        fields=['word'],
                        empty='all')

    strain_chart_items = alt.Chart(
        result_strain_items[lambda df: df['timestep'] == n_timesteps - 1]
    ).add_selection(sel).mark_line(point=True).encode(
        y='sse',
        x='epoch',
        color='word',
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        column='frequency',
        row='pho_consistency',
        tooltip=['epoch', 'word', 'pho', 'output', 'acc', 'sse'])

    strain_chart_items.save(plotsPath + 'strain_chart_items.html')
    strain_chart_items


sel = alt.selection(type='single', on='click', fields=['word'], empty='all')

base = alt.Chart(
    result_grain_items[lambda df: df['epoch'] == nEpo]).add_selection(
        sel).mark_line(point=True).encode(
            x='unit_time',
            color='word',
            column='condition',
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            tooltip=[
                'epoch', 'word', 'output', 'pho_large', 'pho_small',
                'is_large_grain', 'is_small_grain'
            ])

amb = base.encode(y='sse_large_grain').transform_filter(
    (datum.condition == 'ambiguous'))

unamb = base.encode(y='sse_small_grain').transform_filter(
    (datum.condition == 'unambiguous'))

grain_chart_items = amb | unamb
grain_chart_items.save(plotsPath + 'grain_chart_items.html')
grain_chart_items

result_grain_items['unit_time'] = result_grain_items['timestep'] * tau

sel = alt.selection(type='single', on='click', fields=['word'], empty='all')

base = alt.Chart(
    result_grain_items[lambda df: df['epoch'] == nEpo]).add_selection(
        sel).mark_line(point=True).encode(
            y='sse_acceptable',
            x='unit_time',
            color='word',
            opacity=alt.condition(sel, alt.value(1), alt.value(0)),
            tooltip=[
                'epoch', 'word', 'output', 'pho_large', 'pho_small',
                'is_large_grain', 'is_small_grain'
            ])

base

sel = alt.selection(type='single', on='click', fields=['word'], empty='all')

base = alt.Chart(
    result_grain_items[lambda df: df['timestep'] == n_timesteps - 1]
).add_selection(sel).mark_line(point=True).encode(
    y='sse_acceptable',
    x='epoch',
    color='word',
    opacity=alt.condition(sel, alt.value(1), alt.value(0)),
    tooltip=[
        'epoch', 'word', 'output', 'pho_large', 'pho_small', 'is_large_grain',
        'is_small_grain'
    ])

base
