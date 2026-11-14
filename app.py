from pathlib import Path
import sys
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from sky_explorer import search_spherex, build_sequence, load_image_stack, get_display_limits, make_reference_grid

st.set_page_config(page_title='PlanetHunter', page_icon='◉', layout='wide')
LIME = '#C5FF41'
BLACK = '#080808'
WHITE = '#F5F5F5'
st.markdown('''<style>
.stApp, [data-testid="stHeader"] {background:#080808;color:#f5f5f5;}
.block-container {max-width:1450px;padding-top:2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#111111;border-right:1px solid #303030;}
h1,h2,h3 {color:#f5f5f5;letter-spacing:-.8px;}
.stButton button,.stDownloadButton button {border-radius:5px;border:1px solid #444;background:#171717;color:#fff;font-weight:600;}
.stButton button:hover,.stDownloadButton button:hover {border-color:#c5ff41;color:#c5ff41;}
button[kind="primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {background:#c5ff41!important;color:#080808!important;border-color:#c5ff41!important;}
button[kind="primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p {color:#080808!important;}
[data-testid="stMetric"] {background:#111;border:1px solid #303030;padding:16px;border-radius:6px;}
[data-testid="stMetricValue"] {color:#f5f5f5;font-variant-numeric:tabular-nums;}
[data-testid="stMetricLabel"] {color:#aaa;}
[data-testid="stPlotlyChart"] {border:1px solid #303030;border-radius:6px;overflow:hidden;}
.stTabs [data-baseweb="tab-list"] {gap:20px;border-bottom:1px solid #303030;}
.stTabs [aria-selected="true"] {color:#c5ff41!important;}
.stTabs [data-baseweb="tab-highlight"] {background:#c5ff41!important;}
.ph-brand {font-size:13px;font-weight:750;letter-spacing:3px;color:#c5ff41;}
.ph-title {font-size:clamp(2rem,4vw,3.4rem);font-weight:750;letter-spacing:-2px;line-height:1.12;margin:12px 0;}
.ph-sub {color:#aaa;font-size:1rem;max-width:780px;line-height:1.6;}
.ph-hero {padding:12px 0 24px;border-bottom:1px solid #303030;margin-bottom:20px;}
a {color:#c5ff41!important;}
</style>''', unsafe_allow_html=True)

# Reference positions, epoch, proper motion (mas/yr), parallax (mas).
# SIMBAD: Barnard, eps Ind B, RW Aur, R Dra. WISE: A&A 592 A80
# Table 1 (MJD 57123.166); motion/parallax Kirkpatrick et al. 2019.
ASTROMETRY = {
    "Barnard's Star": ('17h57m48.49847s', '+04d41m36.11388s', 2000., -801.551, 10362.394, 546.9759),
    'Epsilon Indi B': ('22h04m10.59518s', '-56d46m58.2391s', 2000., 3981.977, -2466.832, 270.658),
    'WISE 0855': ('08h55m08.163s', '-07d14m39.34s', Time(57123.166, format='mjd').jyear, -8118.9, 679.3, 438.9),
    'RW Aurigae': ('05h07m49.6s', '+30d24m05s', 2000., 7.23, -20.33, 15.42),
    'R Draconis': ('16h32m40.2266s', '+66d45m17.9259s', 2000., -20.643, 6.149, 1.2509),
}


def predicted_position(target, time, parallax=True):
    ra, dec, epoch, pmra, pmdec, plx = ASTROMETRY[target]
    origin = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    dt = time.jyear - epoch
    east, north = pmra*dt/1000, pmdec*dt/1000
    if parallax:
        # Earth proxy for spacecraft position; differential for WISE's observed reference.
        r = get_body_barycentric('earth', time).xyz.to_value(u.au)
        if target == 'WISE 0855':
            r = r - get_body_barycentric('earth', Time(epoch, format='jyear')).xyz.to_value(u.au)
        a, d = origin.ra.rad, origin.dec.rad
        east -= plx/1000 * np.dot(r, [-np.sin(a), np.cos(a), 0])
        north -= plx/1000 * np.dot(r, [-np.cos(a)*np.sin(d), -np.sin(a)*np.sin(d), np.cos(d)])
    return origin.spherical_offsets_by(east*u.arcsec, north*u.arcsec)


TARGETS = {}
for name, fov, detector, description in [
    ("Barnard's Star", .05, 1, 'Nearby high-proper-motion red dwarf.'),
    ('Epsilon Indi B', .05, 5, 'Unresolved brown dwarf binary; distinct from Epsilon Indi A.'),
    ('WISE 0855', .05, 6, 'Very cold brown dwarf; detection is not guaranteed.'),
    ('RW Aurigae', .05, 1, 'Young variable stellar system.'),
    ('R Draconis', .05, 1, 'Long-period Mira variable.'),
]:
    pos = predicted_position(name, Time(2026., format='jyear'), parallax=False)
    TARGETS[name] = (f'{pos.ra.deg:.7f}', f'{pos.dec.deg:.7f}', fov, detector, description)
TARGETS['Custom coordinates'] = ('269.44625', '4.76821', .1, 1, 'Explore any available sky field.')


def target_pixels(target, sequence, ra, dec, fov, shape):
    if target not in ASTROMETRY:
        return [((shape[1]-1)/2, (shape[0]-1)/2)]*len(sequence)
    wcs, grid_shape = make_reference_grid(ra, dec, fov)
    if tuple(grid_shape) != tuple(shape):
        raise ValueError('Registered image dimensions do not match the reference grid.')
    return [tuple(float(v) for v in wcs.world_to_pixel(predicted_position(target, Time(float(mjd), format='mjd')))) for mjd in sequence['mjd']]


@st.cache_data(ttl=3600, show_spinner=False)
def retrieve(ra, dec, fov, detector, epochs):
    cache = ROOT / 'data' / 'demo_cache' / f'{ra:.7f}_{dec:.7f}_{fov:.5f}_D{detector}_N{epochs}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return pd.read_json(StringIO(str(saved['metadata'].item())), orient='split', convert_dates=False), saved['images']
    catalogue = search_spherex(ra, dec, fov)
    sequence = build_sequence(ra, dec, fov, detector=detector, target_epochs=epochs, min_separation_days=14, catalogue=catalogue)
    if sequence is None or len(sequence) == 0:
        raise ValueError('No usable registered observations. Try another detector or field size.')
    images = load_image_stack(sequence)
    # Best-effort portable cache; no original FITS paths required for display.
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        import tempfile, os
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix='.npz', delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            np.savez_compressed(temp_path, images=images, metadata=sequence.to_json(orient='split'))
            os.replace(temp_path, cache)
        finally:
            temp_path.unlink(missing_ok=True)
    except OSError:
        pass
    return sequence, images


def parse_coordinates(ra, dec):
    try:
        r = float(ra)
    except ValueError:
        r = SkyCoord(ra=ra, dec=0*u.deg, unit=(u.hourangle,u.deg)).ra.deg
    try:
        d = float(dec)
    except ValueError:
        d = SkyCoord(ra=0*u.deg, dec=dec, unit=(u.deg,u.deg)).dec.deg
    if not np.isfinite([r,d]).all() or not (0 <= r < 360 and -90 <= d <= 90):
        raise ValueError('RA must be in [0,360) degrees and Dec in [-90,90] degrees.')
    return r,d


def heat(image, lo, hi, smooth, scale='Gray', unit='MJy/sr', showscale=True):
    return go.Heatmap(z=image, zmin=lo, zmax=hi, colorscale=scale,
                      zsmooth='best' if smooth else False, showscale=showscale,
                      colorbar=dict(title=unit, thickness=12),
                      hovertemplate='x: %{x}<br>y: %{y}<br>Value: %{z:.4g}<extra></extra>', hoverongaps=False)


def style_figure(fig, height=600, bottom=35):
    fig.update_layout(template='plotly_dark', paper_bgcolor=BLACK, plot_bgcolor=BLACK,
                      font=dict(color=WHITE, family='Arial', size=12), height=height,
                      margin=dict(l=15,r=15,t=55,b=bottom), showlegend=False,
                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,constrain='domain'),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,scaleanchor='x',scaleratio=1))
    return fig


def reticle(x, y, shape, radius=4):
    h, w = shape
    if not (np.isfinite([x,y]).all() and -.5 <= x <= w-.5 and -.5 <= y <= h-.5):
        return go.Scatter(x=[], y=[], mode='lines', showlegend=False, hoverinfo='skip')
    r, l = radius, radius*.4
    xs = [x-r,x-r,x-r+l,None,x+r-l,x+r,x+r,None,x+r,x+r,x+r-l,None,x-r+l,x-r,x-r]
    ys = [y-r+l,y-r,y-r,None,y-r,y-r,y-r+l,None,y+r-l,y+r,y+r,None,y+r,y+r,y+r-l]
    return go.Scatter(x=xs,y=ys,mode='lines',line=dict(color=LIME,width=2),hoverinfo='skip',showlegend=False)


def image_boundary(fig, shape):
    h,w = shape
    fig.update_xaxes(range=[-.5,w-.5],showline=True,linecolor='#777777',linewidth=1,mirror=True)
    fig.update_yaxes(range=[-.5,h-.5],showline=True,linecolor='#777777',linewidth=1,mirror=True)
    fig.update_layout(plot_bgcolor='#262626', dragmode='zoom')
    fig.add_shape(type='rect',x0=-.5,y0=-.5,x1=w-.5,y1=h-.5,line=dict(color='#777777',width=1),layer='above')


def make_animation(images, sequence, vmin, vmax, speed, colorscale, show_center, smooth_display=True, positions=None, radius=4):
    dates = [str(d)[:10] for d in sequence['date']]
    unit = str(sequence.iloc[0].get('bunit','MJy/sr'))
    h,w = images[0].shape
    if positions is None:
        positions = [((w-1)/2,(h-1)/2)]*len(images)
    def traces(i):
        result = [heat(images[i],vmin,vmax,smooth_display,colorscale,unit)]
        if show_center:
            result.append(reticle(*positions[i], images[i].shape, radius))
        return result
    fig = go.Figure(data=traces(0))
    fig.frames = [go.Frame(name=str(i),data=traces(i),traces=list(range(len(traces(i)))),
                           layout=go.Layout(title_text=f'Observation {i+1:02d} / {len(images):02d}  ·  {dates[i]}')) for i in range(len(images))]
    style_figure(fig, height=650, bottom=105)
    fig.update_layout(title=dict(text=f'Observation 01 / {len(images):02d}  ·  {dates[0]}',x=.02),
        sliders=[dict(active=0,x=.02,len=.96,y=-.06,pad=dict(t=12),currentvalue=dict(prefix='Epoch: ',font=dict(color=LIME)),
                      bgcolor='#303030',activebgcolor=LIME,bordercolor='#555555',
                      steps=[dict(label=d,method='animate',args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i,d in enumerate(dates)])],
        updatemenus=[dict(type='buttons',direction='left',x=.02,y=1.08,xanchor='left',yanchor='bottom',showactive=False,
            bgcolor='#181818',bordercolor='#444444',font=dict(color=WHITE),buttons=[
                dict(label='▶ Play',method='animate',args=[None,dict(fromcurrent=True,frame=dict(duration=speed,redraw=True),transition=dict(duration=0))]),
                dict(label='Ⅱ Pause',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False),transition=dict(duration=0))])])])
    image_boundary(fig, images[0].shape)
    return fig


def static_image(image, lo, hi, title, smooth, scale='Gray', unit='MJy/sr'):
    fig = go.Figure(heat(image,lo,hi,smooth,scale,unit))
    style_figure(fig, height=430)
    fig.update_layout(title=dict(text=title,x=.02))
    image_boundary(fig, image.shape)
    return fig

st.sidebar.markdown('**◉ PLANETHUNTER**')
page = st.sidebar.radio('Navigation',['Sky Explorer','Discovery Lab','About'],label_visibility='collapsed')
st.sidebar.divider()
st.markdown('<div class="ph-hero"><div class="ph-brand">◉ PLANETHUNTER / SPHEREx</div><div class="ph-title">Nothing stands still.</div><div class="ph-sub">Explore a changing universe through real infrared observations from NASA’s SPHEREx mission.</div></div>',unsafe_allow_html=True)

if page == 'Sky Explorer':
    target = st.sidebar.selectbox('Observation target',list(TARGETS))
    ra0,dec0,fov0,d0,description = TARGETS[target]
    st.sidebar.caption(description)
    with st.sidebar.form('field_'+target):
        ra_text = st.text_input('Right ascension',ra0)
        dec_text = st.text_input('Declination',dec0)
        fov = st.slider('Field of view (degrees)',.02,.5,fov0,.01)
        detector = st.selectbox('Detector',list(range(1,7)),index=d0-1,format_func=lambda d:f'D{d}')
        epochs = st.slider('Maximum observations',2,12,6)
        submit = st.form_submit_button('Explore field',type='primary',use_container_width=True)
    if submit:
        try:
            ra,dec = parse_coordinates(ra_text,dec_text)
            with st.spinner('Retrieving and aligning SPHEREx observations…'):
                seq,images = retrieve(ra,dec,fov,detector,epochs)
            st.session_state['sky_data'] = dict(target=target,ra=ra,dec=dec,fov=fov,detector=detector,sequence=seq,images=images)
        except Exception as exc:
            st.error(f'Unable to retrieve this field: {exc}')
    data = st.session_state.get('sky_data')
    if data is None:
        st.subheader('01 / Explore the sky')
        st.write('Choose a target or enter your own coordinates, then select **Explore field** to retrieve real observations.')
        st.caption('Images are retrieved from the NASA/IPAC Infrared Science Archive. The first retrieval may take a minute or longer.')
    else:
        seq,images = data['sequence'],data['images']
        st.subheader(data['target'])
        st.caption(f"RA {data['ra']:.6f}°  ·  Dec {data['dec']:+.6f}°  ·  D{data['detector']}  ·  FOV {data['fov']:.2f}°")
        if target != data['target']:
            st.info('Showing the previously loaded field. Select Explore field to load the new target.')
        a,b,c = st.columns(3)
        a.metric('Observations',f'{len(seq):02d}')
        b.metric('Baseline',f"{seq['mjd'].max()-seq['mjd'].min():.0f} days")
        c.metric('Mean coverage',f"{100*seq['overlap'].mean():.1f}%")
        with st.expander('Display controls'):
            a,b,c = st.columns(3)
            speed = a.slider('Frame duration (ms)',100,2000,700,100)
            upper = b.slider('Upper intensity percentile',90.,100.,99.5,.5)
            colorscale = c.selectbox('Colour map',['Gray','Viridis','Inferno','Magma','Cividis'])
            a,b = st.columns(2)
            smooth = a.checkbox('Smooth image display',True,help='Display interpolation only; does not improve scientific resolution.')
            center = b.checkbox('Show target / field marker',True)
            radius = st.slider('Marker radius (image pixels)',2,10,4)
        vmin,vmax = get_display_limits(images,lower=5,upper=upper)
        tabs = st.tabs(['Time-lapse','Compare observations','Observation metadata'])
        with tabs[0]:
            positions = target_pixels(data['target'],seq,data['ra'],data['dec'],data['fov'],images[0].shape)
            st.plotly_chart(make_animation(images,seq,vmin,vmax,speed,colorscale,center,smooth,positions,radius),use_container_width=True,theme=None,config={'scrollZoom':True,'displaylogo':False})
            st.caption('Common intensity scale. Grey gaps are missing data. Lime brackets show the predicted catalogue position for named targets, or the requested field centre for custom coordinates; not a measured detection.')
            h,w = images[0].shape
            if center and any(not (-.5 <= x <= w-.5 and -.5 <= y <= h-.5) for x,y in positions):
                st.warning('The predicted target is outside the field in one or more epochs. Use the preset coordinates or increase the field of view.')
            if data['target'] in ASTROMETRY:
                with st.expander('Predicted target positions'):
                    st.dataframe(pd.DataFrame({'Date':seq['date'].tolist(),'Predicted x (px)':[p[0] for p in positions],'Predicted y (px)':[p[1] for p in positions]}),hide_index=True)
                    st.caption('Linear proper motion plus approximate Earth-based annual parallax. Spacecraft offset, perspective acceleration and binary orbital motion are neglected. WISE uses an observed 2015 reference position. Predictions do not establish detection.')
            with st.expander('Scientific interpretation'):
                st.write('Images are registered onto a common sky grid. Even within one detector, effective wavelengths may differ. Apparent brightness changes can arise from wavelength, instrumental effects, or genuine variability. Interpolation does not add spatial information.')
        with tabs[1]:
            if len(seq)<2:
                st.info('At least two observations are required for comparison.')
            else:
                a,b = st.columns(2)
                label=lambda i:f"{i+1:02d} · {str(seq.iloc[i]['date'])[:10]}"
                i=a.selectbox('First observation',range(len(seq)),format_func=label)
                j=b.selectbox('Second observation',range(len(seq)),index=len(seq)-1,format_func=label)
                unit=str(seq.iloc[0].get('bunit','MJy/sr'))
                a.plotly_chart(static_image(images[i],vmin,vmax,label(i),smooth,colorscale,unit),use_container_width=True,theme=None)
                b.plotly_chart(static_image(images[j],vmin,vmax,label(j),smooth,colorscale,unit),use_container_width=True,theme=None)
                mode=st.radio('Operation',['Difference (second − first)','Average'],horizontal=True)
                valid=np.isfinite(images[i]) & np.isfinite(images[j])
                result=np.full(images[i].shape,np.nan)
                if mode.startswith('Difference'):
                    result[valid]=images[j][valid]-images[i][valid]
                    lim=float(np.percentile(np.abs(result[valid]),99)) if valid.any() else 1.
                    lim=max(lim,1e-12)
                    fig=static_image(result,-lim,lim,'Difference image',smooth,'RdBu_r',unit)
                else:
                    result[valid]=(images[i][valid]+images[j][valid])/2
                    fig=static_image(result,vmin,vmax,'Average image',smooth,colorscale,unit)
                st.plotly_chart(fig,use_container_width=True,theme=None)
                st.caption('Exploratory comparison only: wavelength differences, PSF variation, noise and registration errors can create residuals.')
        with tabs[2]:
            cols=[c for c in ['epoch_id','date','detector','overlap','em_min_um','em_max_um','bunit'] if c in seq.columns]
            st.dataframe(seq[cols],use_container_width=True,hide_index=True)
            st.download_button('Download observation metadata',seq[cols].to_csv(index=False),'observations.csv','text/csv')
            st.caption('Wavelength bounds describe the archive observation, not necessarily the local effective wavelength at the target.')


elif page == "Discovery Lab":

    st.subheader("02 / Discovery Lab")

    st.write(
        "Search the loaded field for possible "
        "distant moving objects using multi-epoch "
        "astrometry and annual parallax."
    )

    field = st.session_state.get("sky_data")

    if field is None:

        st.info(
            "Load a field in Sky Explorer first, "
            "then return here."
        )

    else:

        st.caption(
            f"{field['target']} · "
            f"RA {field['ra']:.6f}° · "
            f"Dec {field['dec']:+.6f}° · "
            f"D{field['detector']}"
        )

        sequence = field["sequence"]
        images = field["images"]

        a, b, c = st.columns(3)

        a.metric(
            "Available epochs",
            len(sequence)
        )

        b.metric(
            "Field size",
            f"{field['fov']:.2f}°"
        )

        baseline = (
            sequence["mjd"].max()
            - sequence["mjd"].min()
        )

        c.metric(
            "Baseline",
            f"{baseline:.0f} days"
        )

        field_key = (
            field["ra"],
            field["dec"],
            field["fov"],
            field["detector"],
            tuple(sequence["mjd"])
        )

        if st.button(
            "Run moving-object discovery",
            type="primary",
            disabled=len(sequence) < 3
        ):

            st.session_state.pop(
                "live_discovery_result",
                None
            )

            try:

                from live_discovery import (
                    run_discovery
                )

                with st.spinner(
                    "Detecting sources and "
                    "fitting trajectories..."
                ):

                    result = run_discovery(
                        field
                    )

                st.session_state[
                    "live_discovery_result"
                ] = (
                    field_key,
                    result
                )

            except Exception as exc:

                st.error(
                    f"Discovery failed: {exc}"
                )

        if len(sequence) < 3:

            st.info(
                "At least three observations "
                "are required."
            )

        saved = st.session_state.get(
            "live_discovery_result"
        )

        if (
            isinstance(saved, tuple)
            and len(saved) == 2
            and saved[0] == field_key
        ):

            result = saved[1]

            candidates = result[
                "candidates"
            ]

            a, b, c = st.columns(3)

            a.metric(
                "Sources analyzed",
                result["n_sources"]
            )

            b.metric(
                "Candidates",
                len(candidates)
            )

            c.metric(
                "Pass physics screen",
                sum(
                    x["passes_screen"]
                    for x in candidates
                )
            )

            st.caption(
                "Bounded search: up to four "
                "epochs and 20 strongest sources "
                "per epoch. Not an exhaustive survey."
            )

            if candidates:

                columns = [
                    "candidate_id",
                    "n_epochs",
                    "distance_au",
                    "parallax_rms",
                    "linear_rms",
                    "stationary_member_fraction",
                    "escape_speed_ratio",
                    "physics_score",
                    "passes_screen"
                ]

                table = pd.DataFrame(
                    candidates
                )[columns]

                st.dataframe(
                    table,
                    hide_index=True,
                    use_container_width=True
                )

                st.download_button(
                    "Download live candidates",
                    table.to_csv(index=False),
                    "live_candidates.csv",
                    "text/csv"
                )

                st.subheader(
                    "Inspect candidate trajectory"
                )

                i = st.selectbox(
                    "Candidate",
                    range(len(candidates)),
                    format_func=lambda i: (
                        f"{candidates[i]['candidate_id']}"
                        f" · "
                        f"{candidates[i]['distance_au']:.0f}"
                        f" AU"
                    ),
                    key="live_candidate_index"
                )

                candidate = candidates[i]

                from live_discovery import (
                    predict_candidate
                )

                positions = predict_candidate(
                    candidate,
                    field
                )

                lo, hi = get_display_limits(
                    images,
                    lower=5,
                    upper=99.5
                )

                fig = make_animation(
                    images,
                    sequence,
                    lo,
                    hi,
                    700,
                    "Gray",
                    True,
                    True,
                    positions,
                    4
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    theme=None,
                    config={
                        "scrollZoom": True,
                        "displaylogo": False
                    }
                )

                st.caption(
                    "Lime brackets mark the fitted "
                    "trajectory, including predictions "
                    "at unused epochs."
                )

                st.markdown(
                    "**Associated detections**"
                )

                members = pd.DataFrame(
                    candidate["members"]
                )

                st.dataframe(
                    members[[
                        "source_id",
                        "date",
                        "ra",
                        "dec",
                        "peak_snr",
                        "stationary_support"
                    ]],
                    hide_index=True,
                    use_container_width=True
                )

                if (
                    candidate[
                        "stationary_member_fraction"
                    ] >= 0.5
                ):

                    st.warning(
                        "Many associated detections "
                        "are persistent background "
                        "sources. This may be a "
                        "chance association."
                    )

                if (
                    candidate[
                        "escape_speed_ratio"
                    ] > 1
                ):

                    st.warning(
                        "The fitted transverse speed "
                        "exceeds approximate solar "
                        "escape speed at the "
                        "inferred distance."
                    )

                st.warning(
                    "These are unconfirmed candidate "
                    "associations, not planet discoveries. "
                    "Three-epoch fits are weakly constrained."
                )

            else:

                st.info(
                    "No candidate trajectories "
                    "could be fitted."
                )

            with st.expander(
                "Detected source catalogue"
            ):

                st.dataframe(
                    result["sources"],
                    hide_index=True,
                    use_container_width=True
                )

                st.download_button(
                    "Download sources",
                    result["sources"].to_csv(
                        index=False
                    ),
                    "live_sources.csv",
                    "text/csv"
                )

    st.divider()

    st.subheader(
        "Original research experiments"
    )

    enriched = (
        ROOT / "data/enriched_real_candidates.csv"
    )

    ranked = (
        ROOT / "data/ml_ranked_candidates.csv"
    )

    base = (
        ROOT / "data/moving_candidates.csv"
    )

    path = next(
        (
            p for p in (
                enriched,
                ranked,
                base
            )
            if p.exists()
        ),
        None
    )

    if path is not None:

        df = pd.read_csv(path)

        if (
            ranked.exists()
            and "candidate_id" in df
        ):

            ml = pd.read_csv(ranked)

            cols = [
                c for c in (
                    "physics_rank",
                    "ml_rank",
                    "ml_score",
                    "rank_change"
                )
                if c in ml
                and c not in df
            ]

            if cols:

                df = df.merge(
                    ml[["candidate_id"] + cols],
                    on="candidate_id",
                    how="left"
                )

        options = ["Physics ranking"]

        if "ml_rank" in df:

            options.append("ML ranking")

        ranking = st.radio(
            "Research ranking",
            options,
            horizontal=True
        )

        if ranking == "ML ranking":

            df = df.sort_values(
                "ml_rank"
            )

        elif "physics_rank" in df:

            df = df.sort_values(
                "physics_rank"
            )

        else:

            keys = [
                k for k in (
                    "passes_screen",
                    "n_epochs",
                    "physics_score"
                )
                if k in df
            ]

            if keys:

                df = df.sort_values(
                    keys,
                    ascending=[
                        k == "physics_score"
                        for k in keys
                    ]
                )

        cols = [
            k for k in (
                "candidate_id",
                "n_epochs",
                "distance_au",
                "parallax_rms",
                "circular_speed_ratio",
                "stationary_member_fraction",
                "physics_rank",
                "ml_rank",
                "ml_score",
                "passes_screen"
            )
            if k in df
        ]

        st.dataframe(
            df[cols],
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download research candidates",
            df.to_csv(index=False),
            "research_candidates.csv",
            "text/csv"
        )

        st.warning(
            "Research candidates are unconfirmed. "
            "ML scores are not discovery probabilities."
        )

    else:

        st.info(
            "Research candidate files "
            "are not available."
        )

    st.subheader(
        "Synthetic recovery experiment"
    )

    preview = (
        ROOT / "data/synthetic/recovery_preview.png"
    )

    if preview.exists():

        st.image(
            str(preview),
            caption=(
                "Synthetic moving source injected "
                "into real SPHEREx backgrounds."
            )
        )

    truth = (
        ROOT / "data/synthetic/ground_truth.csv"
    )

    if truth.exists():

        st.dataframe(
            pd.read_csv(truth),
            hide_index=True,
            use_container_width=True
        )

    metrics = (
        ROOT / "data/ml/ranking_metrics.csv"
    )

    if metrics.exists():

        st.subheader(
            "Ranking evaluation"
        )

        st.dataframe(
            pd.read_csv(metrics),
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "Synthetic experiments do not establish "
        "independent discovery performance or "
        "survey completeness. The live search "
        "uses physics ranking, not the trained "
        "ML model."
    )


else:

    st.subheader(
        "03 / About PlanetHunter"
    )

    st.write(
        "PlanetHunter retrieves real SPHEREx "
        "infrared observations, aligns them "
        "on a common celestial grid, and "
        "visualizes changes over time."
    )

    st.markdown(
        "**Data:** "
        "[NASA/IPAC IRSA SPHEREx archive]"
        "(https://irsa.ipac.caltech.edu/Missions/spherex.html)"
        " · "
        "[SPHEREx mission]"
        "(https://science.nasa.gov/mission/spherex/)"
    )

    st.markdown(
        "**Methods:** FITS retrieval, WCS "
        "registration, common intensity scaling, "
        "epoch comparison and experimental "
        "motion-model ranking."
    )

    st.markdown(
        "**Limitations:** Detector selection "
        "is not exact wavelength matching. "
        "Predicted positions are not measured "
        "detections. Candidate trajectories "
        "are not confirmed planets."
    )


st.divider()

st.caption(
    "PLANETHUNTER / NASA SPHEREx / "
    "IPAC IRSA / Experimental research prototype"
)




st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <p style="color: #C5FF41; font-size: 14px;">
            Designed & Developed by Dishit Agarwal
        </p>
        <p style="color: #888888; font-size: 12px;">
            Indian Institute of Technology Delhi |
            NASA Space Apps Challenge 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True
)