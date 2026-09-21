"""
IONM MEP Interactive Alarm Dashboard

Visualizes UMAP embedding + alarm features per extremity + timeseries of preprocessed MEPs

Use as standalone:
    python ionm_dashboard.py --run_dir path/to/run_dir

Use from train.py:
    from ionm_dashboard import create_app
    app=create_app(run_dir)
    app.run(debug=False, port=8050)

Required files
    - repr.npy
    - patient_labels.npy
    - side_labels.npy
    - outcome_labels.npy
    - epoch_ids.npy
    - alarm_features.csv

You should open the link that is created in Chrome.
"""

###-------IMPORTING LIBRARIES------###
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Patch

###-------GLOBAL SETTINGS------####
ALARM_THRESHOLD = None # Maybe later: add alarm thresholds to raise an alarm

OUTCOME_COLORS  = {0: '#60BF57', 1: '#CF2626'}
HIGHLIGHT_COLOR = '#000000'

ALARM_FEATURE_COLS = [
    'baseline_dist_euclidean',
    'baseline_dist_manhattan',
    'baseline_dot',
    'consecutive_dist_euclidean',
    'consecutive_dist_manhattan',
    'consecutive_dot'
]

ALARM_FEATURE_LABELS={
    'baseline_dist_euclidean': 'Baseline dist. (Euclidean)',
    'baseline_dist_manhattan': 'Baseline dist. (Manhattan)',
    'baseline_dot': 'Baseline dot product',
    'consecutive_dist_euclidean': 'Consecutive dist. (Euclidean)',
    'consecutive_dist_manhattan': 'Consecutive dist. (Manhattan)',
    'consecutive_dot': 'Consecutive dot product'
}

###-------LOAD DATA------###
def pad_epoch_id(epoch_id):
    parts = str(epoch_id).split('_')
    if len(parts)==2:
        return f'{parts[0]}_{parts[1].zfill(3)}'
    return epoch_id

def load_data(run_dir,mep_data_path, extremity):
    '''
    Function to load data for alarm dashboard.

    Input:
        run_dir         : Folder the dataframes have been saved to
        mep_data_path   : Folder the preprocessed MEPs have been saved to (per muscle)
        extremity (str) : Side you're interested in
    Output:
        master_df (pd.DataFrame)    : Dataframe containing umap coordinates, patient_id, side, outcome, epoch_id
        alarm_df (pd.DataFrame)     : Dataframe containing patient_id, side, outcome, number of meps, summary alarm criteria
        mep_dfs (pd.DataFrame)      : Dataframe containing patient_id, side, outcome, epoch_id, alarm criteria per MEP
    '''

    print('Load data...')
    master_df       = pd.read_csv(os.path.join(run_dir, f'master_df_{extremity}.csv'))
    alarm_df        = pd.read_csv(os.path.join(run_dir, f'alarm_features_{extremity}.csv'))
    mep_alarm_df    = pd.read_csv(os.path.join(run_dir, f'mep_alarm_timeseries_{extremity}.csv'))

    if extremity == 'leg':
        channels = ['QUAD']
    else:
        channels = ['GAS','TA','AH']
    
    mep_dfs = {}
    for ch in channels:
        for side in ['L','R']:
            path = os.path.join(mep_data_path, f'preprocessed_MEP_{ch}_{side}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['id'] = df['id'].apply(pad_epoch_id)
                mep_dfs[f'{ch}_{side}'] = df
            else: 
                print(f'Not found: {path}')
    return master_df, alarm_df, mep_alarm_df, mep_dfs

###------FUNCTION TO CREATE APP -> DASHBOARD-------###
def create_app(run_dir, mep_data_path, extremity):
    '''
    Create dashboard.

    Input:
        run_dir         : Folder the dataframes have been saved to
        mep_data_path   : Folder the preprocessed MEPs have been saved to (per muscle)
        extremity (str) : Side you're interested in
    Output: 
        app
    '''

    # Load data
    master_df, alarm_df, mep_alarm_df, mep_dfs = load_data(run_dir, mep_data_path, extremity)

    patient_side_options = []
    for _, row in alarm_df.iterrows():
        patient = row['patient_id']
        side    = row['side']
        outcome = int(row['outcome'])
        label   = f"{patient} | {side} | {'decline' if outcome==1 else 'no decline'}"
        patient_side_options.append({'label':label, 'value':f"{patient}|{side}"})
    
    # Layout of the dashboard
    app         = dash.Dash(__name__)
    app.title   = 'IONM MEP Alarm Dashboard'

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("IONM MEP Alarm Dashboard",
                style={'margin':'0','color':'black','fontSize':'30px','fontWeight':'600'}),
            html.P("Interactive visualization of encoded representations and alarm features",
                style={'margin':'4px 0 0','color':'#555','fontSize':'22px'}),
        ], style={
            'padding':'16px 24px',
            'borderBottom':'2px solid #5AA5DB',
            'backgroundColor': '#f8fafc',
        }),

        # Dropdown to select patient/side
        html.Div([
            html.Div([
                html.Label('Select patient|side:',
                    style={'fontWeight':'600','fontSize':'22px','marginBottom':'6px'}),
                dcc.Dropdown(
                    id='patient-side-dropdown',
                    options=patient_side_options,
                    value=patient_side_options[0]['value'],
                    clearable=False,
                    style={'fontSize':'22px'}
                ),
            ], style={'flex':'1','marginRight':'24px'}),
   
            # Play and stop button to automatically see the alarm criteria throughout surgery
            html.Div([
                html.Button('Play', id='play-btn',n_clicks=0,
                    style={'backgroundColor':'#5AA5DB','color':'white',
                        'border':'none','borderRadius':'6px',
                        'padding':'8px 16px','cursor':'pointer',
                        'marginRight':'8px','fontsize':'22px'}),
                html.Button('Stop',id='stop-btn',n_clicks=0,
                    style={'backgroundColor':'#888','color':'white',
                        'border':'none','borderRadius':'6px',
                        'padding':'8px 16px','cursor':'pointer',
                        'marginRight':'8px','fontsize':'22px'}),
            ], style={'marginRight':'24px'}),

            # Epoch slider to manually walk through the data
            html.Div([
                html.Label('Epoch slider:', style={'fontWeight':'600','fontSize':'22px', 'marginBottom':'6px'}),
                dcc.Slider(id='epoch-slider',min=0,max=1,step=1,value=0,marks={},
                    tooltip={'placement':'bottom','always_visible':True}),
            ], style={'flex':'2'}),
        ], style={'display':'flex','alignItems':'flex-end','padding':'16px 24px',
                'backgroundColor':'#f8fafc','borderBottom':'1px solid #e0e0e0',
                'gap':'8px'}),

        # Summary stats, representation space, alarm features over time
        html.Div([
            html.Div([
                html.Div(id='summary-stats',style={
                    'fontSize':'18px','height':'40px','display':'flex','alignItems':'center','marginBottom':'16px'
                }),
                html.H3('Representation space (UMAP)',
                    style={'fontSize':'22px','fontWeight':'600','color':'black','marginBottom':'8px'}),
                dcc.Graph(id='umap-plot',
                    style={'height':'80vh'},
                    config={'displayModeBar':False}),
            ], style={'flex':'1', 'padding':'16px','borderRight':'1px solid #e0e0e0'}),
            
            html.Div([
                html.H3('Alarm features over time',
                    style={'fontSize':'22px','fontWeight':'600','color':'black','marginBottom':'8px','height':'40px','display':'flex','alignItems':'center'}),
                dcc.Graph(id='alarm-plots',
                    style={'height':'75vh','width':'100%'},
                    config={'displayModeBar':False}),
            ], style={'flex':'1.2','padding':'16px'}),
        ], style={'display':'flex'}),

        # Preprocessed MEPs over time
        html.Div([
            html.H3('Preprocessed MEPs', style={
                'fontSize':'22px','fontWeight':'600',
                'color':'black','marginBottom':'8px',
                'padding':'16px',
            }),
            dcc.Graph(
                id='mep-overview',
                style={'height':'300px'},
                config={'displayModeBar':False},
            ),
        ], style={'borderTop':'1px solid #e0e0e0','padding':'16px'}),

        dcc.Interval(id='animation-interval',interval=500,n_intervals=0,disabled=True),
        dcc.Store(id='animation-running',data=False),
    ], style={'fontFamily':'Inter,Helvetica,Arial,sans-serif','backgroundColor':'#fff'})

    # Callbacks
    @app.callback(
        Output('epoch-slider','max'),
        Output('epoch-slider','marks'),
        Output('epoch-slider','value'),
        Input('patient-side-dropdown','value'),
    )

    def update_slider(patient_side):
        '''
        Update slider based on selected patient/side
        '''
        patient, side   = patient_side.split('|')
        mask            = (master_df['patient_id']==patient)&(master_df['side']==side)
        n               = int(mask.sum())
        marks           = {}

        if n <=20:
            marks = {i: str(i) for i in range(n)}
        else:
            for i in [0, n//4, n//2, 3*n//4, n-1]:
                marks[str(i)] = str(i)
        return n-1, marks, 0

    @app.callback(
        Output('umap-plot','figure'),
        Input('patient-side-dropdown','value')
    )
    def update_tsne_background(patient_side):
        '''
        Create t-SNE or UMAP background -> changes when a new patient/side is selected
        '''
  
        fig = go.Figure()

        # Background: all other epochs (not selected patient/side)
        fig.add_trace(go.Scatter(
            x=master_df['umap_x'].tolist(),
            y=master_df['umap_y'].tolist(),
            mode='markers',
            marker=dict(
                color=[OUTCOME_COLORS[int(o)] for o in master_df['outcome']],
                size=4, opacity=0.1,
            ),
            text=[f'Epoch {row['epoch_id'][-3:]}<br>{row['patient_id']} | {row['side']}'
                for _, row in master_df.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            name='Other epochs',
            showlegend=False
        ))

        # Gradient coloring of patient/side through UMAP embedding
        fig.add_trace(go.Scatter(
            x=[None],y=[None],
            mode='markers',
            marker = dict(color=[0],colorscale='Viridis',showscale=True,colorbar=dict(title='MEP epoch',thickness=10,len=0.5),cmin=0,cmax=100,size=7,opacity=1.0),
            text=[],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))

        # Black star indicating baseline MEP
        fig.add_trace(go.Scatter(
            x=[None],y=[None],
            mode='markers',
            marker=dict(color='black',size=8,symbol='star',line=dict(color='black',width=1)),
            name='Baseline MEP',
            hovertemplate='Baseline MEP<extra></extra>'
        ))
        
        # Legend
        for outcome_val, color in OUTCOME_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None],y=[None],mode='markers',
                marker=dict(color=color, size=8),
                name='No decline' if outcome_val==0 else 'Motor decline',
            ))
        
        fig.update_layout(
            plot_bgcolor='#f8fafc',paper_bgcolor='#fff',
            margin=dict(l=8, r=8, t=8, b=8),
            legend=dict(orientation='h',y=-0.05,x=0, font=dict(size=16)),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,title='UMAP 1',title_font=dict(size=16)),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,title='UMAP 2',title_font=dict(size=16)),
        )
        return fig

    @app.callback(
        Output('umap-plot','figure',allow_duplicate=True),
        Input('epoch-slider','value'),
        Input('patient-side-dropdown','value'),
        prevent_initial_call='initial_duplicate'
    )

    def update_colored_points(epoch_idx,patient_side):
        patient, side = patient_side.split('|')
        mask_selected = (master_df['patient_id']==patient)&(master_df['side']==side)

        # selected patient / side
        sel_df  = master_df[mask_selected].reset_index(drop=True)
        
        # past epochs -> gradiently colored
        past_df = sel_df.iloc[:epoch_idx+1]

        patched_fig                                 = Patch()
        patched_fig['data'][1]['x']                 = past_df['umap_x'].tolist()
        patched_fig['data'][1]['y']                 = past_df['umap_y'].tolist()
        patched_fig['data'][1]['marker']['cmax']    = int(len(sel_df)-1)
        patched_fig['data'][1]['marker']['color']   = past_df.index.tolist()
        patched_fig['data'][1]['text']              = [f'Epoch {row['epoch_id'][-3:]}<br>{row['patient_id']} | {row['side']}'
                                                        for _,row in past_df.iterrows()]

        patched_fig['data'][2]['x']                 = [float(sel_df.loc[0,'umap_x'])]
        patched_fig['data'][2]['y']                 = [float(sel_df.loc[0,'umap_y'])]
        return patched_fig

    @app.callback(
        Output('alarm-plots','figure'),
        Input('patient-side-dropdown','value'),
        Input('epoch-slider','value'),
    )
    
    def update_alarm_plots(patient_side, epoch_idx):
        '''
        Update alarm plots based on selected patient/side.
        '''

        patient, side = patient_side.split('|')

        sel_full      = mep_alarm_df[(mep_alarm_df['patient_id']==patient) & (mep_alarm_df['side']==side)].reset_index(drop=True)

        # Save for axes
        y_ranges={}
        for feat in ALARM_FEATURE_COLS:
            y_min = float(sel_full[feat].min())
            y_max = float(sel_full[feat].max())
            margin = (y_max-y_min)*0.1 if y_max != y_min else 0.1
            y_ranges[feat] = [y_min - margin, y_max+margin]

        x_max = int(sel_full['epoch_index'].max())

        # Only show alarm criteria for past epochs
        sel = sel_full.iloc[:epoch_idx +1]

        feature_timeseries = {
            'baseline_dist_euclidean':      (sel['baseline_dist_euclidean'].values, sel['epoch_index'].values),
            'baseline_dist_manhattan':      (sel['baseline_dist_manhattan'].values, sel['epoch_index'].values),
            'baseline_dot':                 (sel['baseline_dot'].values, sel['epoch_index'].values),
            'consecutive_dist_euclidean':   (sel['consecutive_dist_euclidean'].values, sel['epoch_index'].values),
            'consecutive_dist_manhattan':   (sel['consecutive_dist_manhattan'].values, sel['epoch_index'].values),
            'consecutive_dot':              (sel['consecutive_dot'].values, sel['epoch_index'].values)
        }

        # Create subplots for alarm criteria over time
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            shared_yaxes=False,
            column_titles=['Comparison with baseline','Comparison with previous epoch'],
            subplot_titles=[
                ALARM_FEATURE_LABELS['baseline_dist_euclidean'],
                ALARM_FEATURE_LABELS['consecutive_dist_euclidean'],
                ALARM_FEATURE_LABELS['baseline_dist_manhattan'],
                ALARM_FEATURE_LABELS['consecutive_dist_manhattan'],
                ALARM_FEATURE_LABELS['baseline_dot'],
                ALARM_FEATURE_LABELS['consecutive_dot']],
            vertical_spacing=0.10,
            horizontal_spacing=0.08
        )

        baseline_feats = ['baseline_dist_euclidean','baseline_dist_manhattan','baseline_dot']
        consecutive_feats = ['consecutive_dist_euclidean','consecutive_dist_manhattan','consecutive_dot']

        for row_idx in range(1,4):
            for col_idx, feat_list in enumerate([baseline_feats, consecutive_feats], start=1):
                feat            = feat_list[row_idx -1]
                y_vals, x_vals  = feature_timeseries[feat]

                fig.add_trace(go.Scatter(
                    x=x_vals.tolist(), y=y_vals.tolist(),
                    mode='lines',
                    line=dict(color='black',width=2),
                    showlegend=False,
                ), row=row_idx, col=col_idx)

            fig.update_xaxes(range=[0,x_max],row=row_idx,col=1,showgrid=True,gridcolor='#e0e0e0',gridwidth=1)
            fig.update_xaxes(range=[0,x_max],row=row_idx,col=2,showgrid=True,gridcolor='#e0e0e0',gridwidth=1)
            fig.update_yaxes(range=y_ranges[baseline_feats[row_idx-1]],row=row_idx, col=1, showgrid=True, gridcolor='#e0e0e0',gridwidth=1)
            fig.update_yaxes(range=y_ranges[consecutive_feats[row_idx-1]],row=row_idx, col=2, showgrid=True, gridcolor='#e0e0e0',gridwidth=1)

        # Update layout 
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=40,r=40,t=80,b=40),
            height=950,
            font=dict(size=14)
        )

        for annotation in fig.layout.annotations:
            if annotation.text in ['Comparison with baseline','Comparison with previous epoch']:
                annotation.y = annotation.y+0.05
                annotation.font = (dict(size=22, weight='bold',color='black'))
            else:
                annotation.font = (dict(size=18,color='black'))

        fig.update_xaxes(title_text='Epoch index',row=3, col=1,showgrid=True,gridcolor='#e0e0e0',title_font=dict(size=16))
        fig.update_xaxes(title_text='Epoch index',row=3, col=2,showgrid=True,gridcolor='#e0e0e0',title_font=dict(size=16))
        return fig
    
    @app.callback(
        Output('mep-overview','figure'),
        Input('epoch-slider','value'),
        Input('patient-side-dropdown','value'),
    )
    def update_mep_overview(epoch_idx, patient_side):
        '''
        Update preprocessed MEPs on the bottom of the dashboard
        '''

        if patient_side is None or '|' not in str(patient_side):
            raise dash.exceptions.PreventUpdate
        
        patient, side   = patient_side.split('|')
        side_letter     = 'L' if side == 'left' else 'R'

        epoch_idx       = int(epoch_idx)
        
        if extremity == 'leg':
            channels=['QUAD']
        else:
            channels=['GAS','TA','AH']

        # Create subplots: 1 for leg, 3 for foot
        n_channels  = len(channels)
        fig         = make_subplots(
            rows=n_channels, cols=1,
            shared_xaxes=True,vertical_spacing=0.05,subplot_titles=channels
        )

        # Determine which epochs to show: current +- 2
        sel             = master_df[(master_df['patient_id']==patient)&(master_df['side']==side)].reset_index(drop=True)
        n_total         = len(sel)
        indices_to_show = [i for i in range(epoch_idx -2, epoch_idx +3) if 0 <=i < n_total]
        colors          = {-2: '#cccccc',-1:'#999999',0:'#1a1a1a',1:'#999999',2:'#cccccc'}
        linewidths      = {-2:1, -1:1, 0:2, 1:1, 2:1}

        # Fill in subplots with preprocessed MEPs
        for row_idx, ch in enumerate(channels, start=1):
            key = f'{ch}_{side_letter}'
            if key not in mep_dfs:
                continue

            ch_df  = mep_dfs[key]
            offset = 0

            for pos, ep_idx in enumerate(indices_to_show):
                epoch_id = sel.loc[ep_idx,'epoch_id']
                ep_data  = ch_df[ch_df['id']==epoch_id].sort_values('time')

                if len(ep_data) == 0:
                    continue

                values      = ep_data['value_scaled'].values
                n_samples   = len(values)
                x_vals      = list(range(offset, offset+n_samples))
                rel_pos     = ep_idx - epoch_idx

                is_current  = rel_pos == 0

                fig.add_trace(go.Scatter(
                    x=x_vals,y=values.tolist(),mode='lines',
                    line=dict(color=colors.get(rel_pos,'#999999'),
                    width=linewidths.get(rel_pos,1),
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                ), row=row_idx,col=1)

                # Use dotted lines to separate MEPs
                if pos<len(indices_to_show)-1:
                    fig.add_vline(
                        x=offset+n_samples-0.5,
                        line=dict(color='gray',width=1,dash='dot'),
                        row=row_idx, col=1
                    )
                offset += n_samples
            fig.update_yaxes(
                showgrid=False,
                zeroline=True,
                zerolinecolor='lightgray',
                zerolinewidth=1,
                row=row_idx,col=1
            )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=40,r=20,t=30,b=20),
            height=300,
        )
        fig.update_xaxes(showgrid=False,showticklabels=False)
        return fig

    @app.callback(
        Output('summary-stats','children'),
        Input('patient-side-dropdown','value'),
    )
    def update_summary(patient_side):
        '''
        Update summary metrics
        '''

        patient, side = patient_side.split('|')

        # Find information for selected patient/side
        row     = alarm_df[(alarm_df['patient_id']==patient)&(alarm_df['side']==side)].iloc[0]
        netd    = float(row['net_displacement'])
        outcome = int(row['outcome'])

        return html.Div([
            html.H3('Summary', style={'fontSize':'22px','fontWeight':'600','marginBottom':'8px'}),
            html.Span(f'Net displacement: {netd:.2f}',style={'marginRight':'32px'}),
            html.Span(f'Outcome: {'motor decline' if outcome ==1 else 'No decline'}',
                style={'color':'#CF2626' if outcome ==1 else '#60BF57','fontWeight':'600'}),
        ])

    @app.callback(
        Output('animation-interval','disabled'),
        Output('animation-running','data'),
        Input('play-btn','n_clicks'),
        Input('stop-btn','n_clicks'),
        State('animation-running','data'),
        prevent_initial_call=True,
    )
    def toggle_animation(play_clicks, stop_clicks, is_running):
        ctx = dash.callback_context
        if not ctx.triggered:
            return True, False
        triggered = ctx.triggered[0]['prop_id']
        if 'play-btn' in triggered:
            return False, True
        else:
            return True, False

    @app.callback(
        Output('epoch-slider','value', allow_duplicate=True),
        Input('animation-interval','n_intervals'),
        State('epoch-slider','value'),
        State('epoch-slider','max'),
        State('animation-running','data'),
        prevent_initial_call=True
    )

    def advance_epoch(n_intervals, current_val, max_val, is_running):
        if not is_running:
            raise dash.exceptions.PreventUpdate
        current_val=int(current_val)
        max_val=int(max_val)
        next_val=current_val+1
        if next_val>max_val:
            return max_val
        return next_val
    
    @app.callback(
        Output('animation-interval','disabled',allow_duplicate=True),
        Output('animation-running','data',allow_duplicate=True),
        Input('epoch-slider','value'),
        State('epoch-slider','max'),
        State('animation-running','data'),
        prevent_initial_call=True
    )

    def stop_at_end(current_val,max_val,is_running):
        if is_running and int(current_val)>=int(max_val):
            return True, False
        raise dash.exceptions.PreventUpdate
    return app

# STANDALONE use
if __name__ == '__main__':
    import argparse
    parser    = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,required=True, help='dataset')
    parser.add_argument('--run-dir',type=str, required=True,help='folder with all files')
    parser.add_argument('--port',type=int,default=8050)
    args_dash = parser.parse_args()

    app       = create_app(args_dash.run_dir, args.dataset)
    print(f'Dashboard on http://localhost:{args_dash.port}')
    app.run(debug=False,port=args_dash.port)
