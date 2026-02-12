import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean

from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

from inversion_sst_gp.utils import compute_good_data_over_time
from inversion_sst_gp.plot_helper import draw_confidence_ellipse

mpl.rcParams.update({
    "figure.autolayout": False,   # critical: you control margins
    "axes.linewidth": 1.5,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,

    "figure.subplot.left": 0.12,
    "figure.subplot.right": 0.96,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.top": 0.85,
})

def plot_gradients(ds, tg_name='dTdt', sg_names=['dTdx','dTdy']):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace':0.25})

    ds[tg_name].plot(ax=ax[0], cmap='viridis', vmin=0, vmax = 4e-5, cbar_kwargs={'pad':0.03, 'shrink':0.5, 'label':'$\\partial T/ \\partial t$ [K s$^{-1}$]'})

    np.abs(ds[sg_names[0]] + 1j*ds[sg_names[1]]).plot(ax=ax[1], cmap='Reds', vmin=0, vmax=1e-4, cbar_kwargs={'pad':0.03, 'shrink':0.5, 'label':'$\\partial T/ \\partial s$ [K m$^{-1}$]'})

    ax[0].set_title('Temporal gradient')
    ax[1].set_title('Spatial gradient')

    for x in ax:
        x.set_aspect('equal')
        x.set_xlabel('')
        x.set_ylabel('')
        
    return fig, ax


def plot_scene(ds, T_name='T', u_name='mu_u', v_name='mu_v', ax=None, qv_scale=20, qk_size=1.0, qk_x=0.78, qk_y=1.03, cbar=True, vlims=None):
    
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        return_fig = False

    if cbar:
        if vlims is None:
            ds[T_name].plot(ax=ax, cmap='cmo.thermal', cbar_kwargs={'label':'SST [K]', 'shrink':0.7, 'pad':0.03})
        else:
            ds[T_name].plot(ax=ax, cmap='cmo.thermal', vmin=vlims[0], vmax=vlims[1], cbar_kwargs={'label':'SST [K]', 'shrink':0.7, 'pad':0.03})
    else:
        if vlims is None:
            ds[T_name].plot(ax=ax, cmap='cmo.thermal', add_colorbar=False)
        else:
            ds[T_name].plot(ax=ax, cmap='cmo.thermal', vmin=vlims[0], vmax=vlims[1], add_colorbar=False)

    ax.set_aspect('equal')
    if 'time' in ds.coords:
        ax.set_title(ds.time.astype('datetime64[h]').values)
    ax.set_xlabel('')
    ax.set_ylabel('')
        
    if u_name is not None:
        Q = ax.quiver(ds['LON'], ds['LAT'], ds[u_name], ds[v_name], scale=qv_scale)
        qk = ax.quiverkey(
                    Q,
                    X=qk_x, Y=qk_y,      
                    U=qk_size,              
                    label=f'{qk_size:.1f}' + 'm s$^{-1}$',
                    labelpos='E',
                    coordinates='axes')
    if return_fig:
        return fig, ax
    else:
        return ax


def plot_prediction(ds, T_name='T', u_name='mu_u', v_name='mu_v', std_u_name='std_u', std_v_name='std_v', qv_scale=20, qk_size=1.0, qk_x=0.78, qk_y=1.03):
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace':0.25})

    ds[T_name].plot(ax=ax[0], cmap='cmo.thermal', cbar_kwargs={'label':'SST [K]', 'shrink':0.7, 'pad':0.03})
    Q = ax[0].quiver(ds['LON'], ds['LAT'], ds[u_name], ds[v_name], scale=qv_scale)

    vm = np.max([np.abs(ds[std_u_name]).max(), np.abs(ds[std_v_name]).max()])
    ds[std_u_name].plot(ax=ax[1], cmap='Greys', vmin=0, vmax=vm, cbar_kwargs={'label':'Uncertainty in $u$ [m s$^{-1}$]', 'shrink':0.7, 'pad':0.03})
    ds[std_v_name].plot(ax=ax[2], cmap='Greys', vmin=0, vmax=vm, cbar_kwargs={'label':'Uncertainty in $v$ [m s$^{-1}$]', 'shrink':0.7, 'pad':0.03})

    for x in ax:
        x.set_aspect('equal')
        x.set_title('\u2007\u2007\u2007')
        x.set_xlabel('')
        x.set_ylabel('')
        
    qk = ax[0].quiverkey(
                    Q,
                    X=qk_x, Y=qk_y,      
                    U=qk_size,              
                    label=f'{qk_size:.1f}' + 'm s$^{-1}$',
                    labelpos='E',
                    coordinates='axes')

    return fig, ax


def plot_mean_function(ds):
    
    if 'mu_u_prior' in ds.data_vars.keys():
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={'wspace':0.15})

        ds['mu_u_prior'].plot(ax=ax[0], cbar_kwargs={'pad':0.01, 'label':'', 'shrink':0.75})
        ds['mu_v_prior'].plot(ax=ax[1], cbar_kwargs={'pad':0.01, 'label':'', 'shrink':0.75})
        ds['mu_S_prior'].plot(ax=ax[2], cbar_kwargs={'pad':0.01, 'label':'', 'shrink':0.75})

        for x,tt in zip(ax, ['$u_{prior}$ [m s$^{-1}$]', '$v_{prior}$ [m s$^{-1}$]', '$S_{prior}$ [K s$^{-1}$]']):
            x.set_title(tt)
            x.set_xlabel('')
            x.set_ylabel('')
            x.set_aspect('equal')
        return fig, ax
        
    else:
        raise Exception('Prior variables not in supplied dataset')
        


def plot_timeseries(ds, lon_pt, lat_pt, u_name='mu_u', v_name='mu_v', std_u_name='std_u', std_v_name='std_v'):
    """Plot timeseries of a variable at a specific point."""

    ds_point = ds.sel(lon=lon_pt, lat=lat_pt, method='nearest')

    fig, ax = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'hspace':0.05})

    ds_point[u_name].plot(ax=ax[0], label='u velocity', c='k')
    ax[0].fill_between(ds['time'], ds_point[u_name] - ds_point[std_u_name],
                       ds_point[u_name] + ds_point[std_u_name], color='k', alpha=0.2)

    ds_point[v_name].plot(ax=ax[1], label='v velocity', c='navy')
    ax[1].fill_between(ds['time'], ds_point[v_name] - ds_point[std_v_name],
                       ds_point[v_name] + ds_point[std_v_name], color='navy', alpha=0.2)

    for x in ax:
        x.set_xlabel('')
        x.set_title('')
        x.set_ylim(-0.8, 0.8)
        x.grid()
        x.set_xlim(ds_point['time'].values[0], ds_point['time'].values[-1])

    ax[0].set_xticklabels([])
    ax[0].set_ylabel('u [m s$^{-1}$]')
    ax[1].set_ylabel('v [m s$^{-1}$]')
    
    return fig, ax



def plot_pred_ellipses(ds, Kpp, n_std=1, scale=1, qv_scale=20, qk_size=1.0, qk_x=0.78, qk_y=1.03, an=True, **kwargs):

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'wspace':0.1})
    
    ds['T'].plot(ax=ax[0], cmap='cmo.thermal', add_colorbar=False)
    Q = ax[0].quiver(ds['LON'], ds['LAT'], ds['mu_u'], ds['mu_v'], scale=qv_scale)
        
    draw_confidence_ellipse(ax[1], ds['LON'], ds['LAT'], Kpp, n_std=n_std, scale=scale, an=an, **kwargs)
    ax[1].set_yticklabels('')

    for x in ax:
        x.set_aspect('equal')
        x.set_title('\u2007\u2007\u2007')
        x.set_xlabel('')
        x.set_ylabel('')
        x.set_xlim(ds['LON'].min(), ds['LON'].max())
        x.set_ylim(ds['LAT'].min(), ds['LAT'].max())

    qk = ax[0].quiverkey(
                    Q,
                    X=qk_x, Y=qk_y,      
                    U=qk_size,              
                    label=f'{qk_size:.1f}' + 'm s$^{-1}$',
                    labelpos='E',
                    coordinates='axes')       
    return fig, ax



def plot_param_series(ds_params, ds_data=None, vel_var='mu_u', vel_idx=(0,0)):
    p_vars = ['sigma_u', 'l_u', 'tau_u', 'sigma_S', 'l_S', 'tau_S']
    y_labels = ['$\\sigma_{\\mathbf{u}}$\n[m s$^{-1}$]', '$\\mathbf{u}$ length\n[km]', '$\\mathbf{u}$ noise\n[m s$^{-1}$]', '$\\sigma_{S}$\n[°C]', '$S$ length\n[km]', '$S$ noise\n[°C]']

    if ds_data is not None:
        fig, ax = plt.subplots(7, 1, figsize=(10, 10), gridspec_kw={'hspace':0.05})

        good_T = compute_good_data_over_time(ds_data, 'T')
        (good_T).plot.scatter(x='time', ax=ax[0], c='red', s=5, label='Good data fraction')
    
        ds_data[vel_var].isel(lon=vel_idx[0], lat=vel_idx[1]).plot(ax=ax[0], subplot_kws={'marker':'.'})
        ax[0].set_ylabel(f'{vel_var} [m s$^{{-1}}$]')
        ax[0].set_xlim(ds_params.time[0], ds_params.time[-1])
        ax[0].set_xticklabels([])
        
        pxx = 1
        
    else:
        fig, ax = plt.subplots(6, 1, figsize=(10, 8.5), gridspec_kw={'hspace':0.05})
        pxx = 0
    
    nanx = np.isnan(ds_params[p_vars[0]])

    for ii, var in enumerate(p_vars):
        ds_params[var].where(~nanx).plot(ax=ax[ii+pxx])
        ax[ii+pxx].plot([ds_params.time.values[0], ds_params.time.values[-1]],\
            [ds_params[var].where(~nanx).median(), ds_params[var].where(~nanx).median()],\
                c='grey', ls=':')
        ax[ii+pxx].set_ylabel(y_labels[ii])
        ax[ii+pxx].set_xlim(ds_params.time[0], ds_params.time[-1])
        ax[ii+pxx].set_title('')
        if ax[ii+pxx] != ax[-1]:
            ax[ii+pxx].set_xticklabels([])
            ax[ii+pxx].set_xlabel('')
            
    return fig, ax



def plot_data_animation(ds, T_name='T', tg_name='dTdt', sg_names=['dTdx','dTdy'], anim_interval=500):
    """
    Create an animation of the GP regression predictions over time.

    Parameters:
        ds (xarray.Dataset): Dataset containing SST data and GP regression results.
        T_name (str): Name of the SST variable in the dataset.
        dT_name (str): Name of the spatial derivative variable in the dataset.
        qk_size (float): Size of the quiver arrows.
        qk_x (float): X position of the quiver key.
        qk_y (float): Y position of the quiver key.
        anim_interval (int): Interval between frames in milliseconds.

    Returns:
        matplotlib.animation.FuncAnimation: Animation object.
    """
    # Create the figure and axes
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.1})

    vm, vx = np.nanmin(ds[T_name].where(ds[T_name] != 0.)), np.nanmax(ds[T_name])
    im0 = ds.isel(time=0)[T_name].plot(ax=ax[0], cmap=cmocean.cm.thermal,\
                                        # vmin=vm, vmax=vx, norm=Normalize(vmin=vm, vmax=vx),\
                                        cbar_kwargs={'pad':0.03, 'shrink':0.5, 'label':'T [$^{\circ}$C]'})
    
    imdt = ds.isel(time=0)[tg_name].plot(ax=ax[1], cmap='viridis',\
                                        vmin=0, vmax = 4e-5,\
                                        cbar_kwargs={'pad':0.03, 'shrink':0.5, 'label':'$\\partial T/ \\partial t$ [K s$^{-1}$]'})

    imds = np.abs(ds.isel(time=0)[sg_names[0]] + 1j*ds.isel(time=0)[sg_names[1]]).plot(ax=ax[2], cmap='Reds',\
        vmin=0, vmax=1e-4, cbar_kwargs={'pad':0.03, 'shrink':0.5, 'label':'$\\partial T/ \\partial s$ [K m$^{-1}$]'})

    ax[1].set_title('Temporal gradient')
    ax[2].set_title('Spatial gradient')

    for x in ax:
        x.set_aspect('equal')
        x.set_xlabel('')
        x.set_ylabel('')
        if x != ax[0]:
            x.set_yticklabels([])

    # Function to update the animation for each frame
    def update(frame):
        # Update SST plot
        im0.set_array(ds.isel(time=frame)[T_name].values.flatten())
        
        # Dynamically update the colormap limits based on the frame
        new_vmin = np.nanmin(ds.isel(time=frame)[T_name])
        new_vmax = np.nanmax(ds.isel(time=frame)[T_name])
        im0.set_clim(vmin=new_vmin, vmax=new_vmax)
        
        # Update the time gradient plot
        imdt.set_array(ds.isel(time=frame)[tg_name].values.flatten())
        
        # Dynamically update the colormap limits based on the frame
        new_vmin = np.nanmin(ds.isel(time=frame)[tg_name])
        new_vmax = np.nanmax(ds.isel(time=frame)[tg_name])
        imdt.set_clim(vmin=new_vmin, vmax=new_vmax)        
        
        # Update the spatial gradient plot
        imds.set_array(np.abs(ds.isel(time=frame)[sg_names[0]] + 1j*ds.isel(time=frame)[sg_names[1]]).values.flatten())
        
        # Dynamically update the colormap limits based on the frame
        new_vmin = np.nanmin(np.abs(ds.isel(time=frame)[sg_names[0]] + 1j*ds.isel(time=frame)[sg_names[1]]))
        new_vmax = np.nanmax(np.abs(ds.isel(time=frame)[sg_names[0]] + 1j*ds.isel(time=frame)[sg_names[1]]))
        imds.set_clim(vmin=new_vmin, vmax=new_vmax) 

        # Update the title of the middle axis with the time
        time_str = str(ds['time'].isel(time=frame).values)
        ax[0].set_title(f"Time: {time_str}")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(ds['time']), interval=anim_interval)

    return ani


def plot_prediction_animation(ds, T_name='T', u_name='mu_u', v_name='mu_v', std_u_name='std_u', std_v_name='std_v',\
                              qv_scale=20, qk_size=1.0, qk_x=0.78, qk_y=1.03, anim_interval=500, smoothing=0.05, add_cbar=True):
    """
    Create an animation of the GP regression predictions over time.

    Parameters:
        ds (xarray.Dataset): Dataset containing SST data and GP regression results.
        T_name (str): Name of the SST variable in the dataset.
        u_name (str): Name of the zonal current mean variable in the dataset.
        v_name (str): Name of the meridional current mean variable in the dataset.
        std_u_name (str): Name of the zonal current standard deviation variable in the dataset.
        std_v_name (str): Name of the meridional current standard deviation variable in the dataset.
        qk_size (float): Size of the quiver arrows.
        qk_x (float): X position of the quiver key.
        qk_y (float): Y position of the quiver key.
        anim_interval (int): Interval between frames in milliseconds.

    Returns:
        matplotlib.animation.FuncAnimation: Animation object.
    """
    # Create the figure and axes
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.25})
    
    vm, vx = np.nanmin(ds[T_name].where(ds[T_name] != 0.)), np.nanmax(ds[T_name])
    vm_std, vx_std = 0., 0.2
    cbar_kwargs = {'pad':0.03, 'shrink':0.5, 'label':'T [$^{\circ}$C]'}
    
    cmap = plt.get_cmap('cmo.thermal').copy()
    cmap.set_bad(color='white')

    # Initialize the plots
    im0 = ds.isel(time=0)[T_name].plot(ax=ax[0], cmap=cmap, add_colorbar=add_cbar, cbar_kwargs=cbar_kwargs)#, vmin=vm, vmax=vx, norm=Normalize(vmin=vm, vmax=vx))
    Q = ax[0].quiver(ds.isel(time=0)['LON'], ds.isel(time=0)['LAT'], 
                    ds.isel(time=0)[u_name], ds.isel(time=0)[v_name],
                    scale=qv_scale)
    
    cbar_kwargs = {'pad':0.03, 'shrink':0.5, 'label':'$U_{std}$ [m s$^{-1}$]'}
    im1 = ds.isel(time=0)[std_u_name].plot(ax=ax[1], cmap='Greys', add_colorbar=add_cbar, vmin=vm_std, vmax=vx_std, cbar_kwargs=cbar_kwargs)
    cbar_kwargs = {'pad':0.03, 'shrink':0.5, 'label':'$V_{std}$ [m s$^{-1}$]'}
    im2 = ds.isel(time=0)[std_v_name].plot(ax=ax[2], cmap='Greys', add_colorbar=add_cbar, vmin=vm_std, vmax=vx_std, cbar_kwargs=cbar_kwargs)

    # Set axis properties
    for x in ax:
        x.set_aspect('equal')
        x.set_title('')
        x.set_xlabel('')
        x.set_ylabel('')

    # initialize smoothed color limits (start from frame 0 / global values)
    current_vmin = np.nanmin(ds.isel(time=0)[T_name]) if not np.isnan(np.nanmin(ds.isel(time=0)[T_name])) else vm
    current_vmax = np.nanmax(ds.isel(time=0)[T_name]) if not np.isnan(np.nanmax(ds.isel(time=0)[T_name])) else vx

    # Function to update the animation for each frame
    def update(frame):
        nonlocal current_vmin, current_vmax
        # Update SST plot
        im0.set_array(ds.isel(time=frame)[T_name].values.flatten())

        # frame extremes
        frame_vmin = np.nanmin(ds.isel(time=frame)[T_name])
        frame_vmax = np.nanmax(ds.isel(time=frame)[T_name])

        # update smoothed limits using exponential moving average
        # small `smoothing` -> slower change; values outside are allowed
        if not np.isnan(frame_vmin):
            current_vmin = (1.0 - smoothing) * current_vmin + smoothing * frame_vmin
        if not np.isnan(frame_vmax):
            current_vmax = (1.0 - smoothing) * current_vmax + smoothing * frame_vmax

        im0.set_clim(vmin=current_vmin, vmax=current_vmax)
        
        # Update quiver plot
        Q.set_UVC(ds.isel(time=frame)[u_name], ds.isel(time=frame)[v_name])
        
        # Update uncertainty plots
        im1.set_array(ds.isel(time=frame)[std_u_name].values.flatten())
        im2.set_array(ds.isel(time=frame)[std_v_name].values.flatten())
        
        # Update the title of the middle axis with the time
        time_str = str(ds['time'].isel(time=frame).values)
        ax[1].set_title(f"Time: {time_str}")

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(ds['time']), interval=anim_interval)

    return ani