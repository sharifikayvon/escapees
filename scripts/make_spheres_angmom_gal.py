import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm
import astropy.coordinates as apycoords
import astropy.units as u
from pathlib import Path


def galcen_cyl_pos(ra, dec, parallax):
    """
    Converts ICRS ra (deg), dec (deg), parallax (mas)
    to Galactocentric cylindrical coordinates
    rho (kpc), phi (deg), z_cyl (pc)

    LEFT-HANDED

    Args:
        ra (array-like): Right ascension in degrees
        dec (array-like): Declination in degrees
        parallax (array-like): Parallax in milliarcseconds

    Returns:
        numpy.ndarray: (n x 3) array of galactocentric cylindrical coordinates


    """

    c = apycoords.SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=(1000.0 / parallax) * u.pc,
        frame="icrs",
    )

    gc_frame = apycoords.Galactocentric(galcen_distance=8.25 * u.kpc, z_sun=20.8 * u.pc)

    gc = c.transform_to(gc_frame)
    gc.representation_type = "cylindrical"

    cyl_coord = np.vstack(
        [
            gc.rho.to(u.kpc).value,
            180 - gc.phi.degree,  # 180 - PHI ASTROPY CONVENTION
            gc.z.to(u.pc).value,
        ]
    ).T

    return cyl_coord


def icrs_to_galactic_pm(ra, dec, pmra, pmdec, parallax=None, distance=None):
    """
    Convert ICRS (pmra, pmdec) to Galactic (pm_l_cosb, pm_b).

    Parameters
    ----------
    ra, dec : float or array-like
        ICRS coordinates in degrees.
    pmra, pmdec : float or array-like
        Proper motions in ICRS frame (mas/yr).
        pmra is pm_ra_cosdec (i.e. already cos(dec)-corrected).
    parallax : float or array-like, optional
        Parallax in mas. Used to compute distance if `distance` not given.
    distance : float or array-like, optional
        Distance in pc. If None, derived from parallax.

    Returns
    -------
    pm_l_cosb, pm_b : ndarray
        Galactic proper motions in mas/yr.
    """
    if distance is not None:
        dist = np.asarray(distance) * u.pc
    elif parallax is not None:
        dist = (1000.0 / np.asarray(parallax)) * u.pc
    else:
        raise ValueError("Must provide either parallax or distance.")

    c = apycoords.SkyCoord(
        ra=np.asarray(ra) * u.deg,
        dec=np.asarray(dec) * u.deg,
        distance=dist,
        pm_ra_cosdec=np.asarray(pmra) * u.mas / u.yr,
        pm_dec=np.asarray(pmdec) * u.mas / u.yr,
        frame="icrs",
    )

    gal = c.galactic

    pm_l_cosb = gal.pm_l_cosb.to(u.mas / u.yr).value
    pm_b = gal.pm_b.to(u.mas / u.yr).value

    return pm_l_cosb, pm_b


def skycoord_cylvel_ang_mom_to_dvT(
    ra, dec, parallax, pmra, pmdec, v_rho, v_phi, v_z, rho
):

    v_sun = apycoords.CartesianDifferential([11.1, 245.0, 7.25] * u.km / u.s)

    gc_frame = apycoords.Galactocentric(
        galcen_distance=8.25 * u.kpc, z_sun=20.8 * u.pc, galcen_v_sun=v_sun
    )

    c = apycoords.SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=(1000.0 / parallax) * u.pc,
        frame="icrs",
    )

    gc = c.transform_to(gc_frame)

    cyl = gc.represent_as(apycoords.CylindricalRepresentation)
    phi = cyl.phi

    v_rho = np.asarray(v_rho) * u.km / u.s
    v_phi = -np.asarray(v_phi) * u.km / u.s
    rho = np.asarray(rho) * u.kpc
    ang_mom = rho * v_phi

    v_z = np.asarray(v_z) * u.km / u.s

    rho_stars = galcen_cyl_pos(ra, dec, parallax)[:, 0] * u.kpc

    v_phi_pred = (ang_mom / rho_stars).to(u.km / u.s)

    vx = v_rho * np.cos(phi) - v_phi_pred * np.sin(phi)
    vy = v_rho * np.sin(phi) + v_phi_pred * np.cos(phi)
    vz = v_z

    gc_vel = apycoords.Galactocentric(
        x=gc.x,
        y=gc.y,
        z=gc.z,
        v_x=vx,
        v_y=vy,
        v_z=vz,
        galcen_distance=8.25 * u.kpc,
        z_sun=20.8 * u.pc,
        galcen_v_sun=v_sun,
    )

    icrs = apycoords.SkyCoord(gc_vel).transform_to(apycoords.ICRS())
    dist = (1000.0 / np.asarray(parallax)) * u.pc

    # ── ICRS residuals ───────────────────────────────────────────────────────
    vra = (
        (np.asarray(pmra) * u.mas / u.yr * dist)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    vdec = (
        (np.asarray(pmdec) * u.mas / u.yr * dist)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )

    proj_vra = (
        (icrs.pm_ra_cosdec * icrs.distance)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    proj_vdec = (
        (icrs.pm_dec * icrs.distance)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    proj_vrad = icrs.radial_velocity.to(u.km / u.s).value

    delta_vra = vra - proj_vra
    delta_vdec = vdec - proj_vdec
    delta_vT = np.sqrt(delta_vra**2 + delta_vdec**2)

    # ── Galactic proper motions & residuals ──────────────────────────────────
    pm_l_cosb_obs, pm_b_obs = icrs_to_galactic_pm(
        ra, dec, pmra, pmdec, parallax=parallax
    )

    vl_obs = (
        (pm_l_cosb_obs * u.mas / u.yr * dist)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    vb_obs = (
        (pm_b_obs * u.mas / u.yr * dist)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )

    gal_pred = icrs.galactic
    proj_vl = (
        (gal_pred.pm_l_cosb * gal_pred.distance)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )
    proj_vb = (
        (gal_pred.pm_b * gal_pred.distance)
        .to(u.km / u.s, equivalencies=u.dimensionless_angles())
        .value
    )

    delta_vl = vl_obs - proj_vl
    delta_vb = vb_obs - proj_vb
    delta_vT_gal = np.sqrt(delta_vl**2 + delta_vb**2)

    return np.vstack(
        [
            vra,
            vdec,
            proj_vra,
            proj_vdec,
            proj_vrad,
            delta_vra,
            delta_vdec,
            delta_vT,
            pm_l_cosb_obs,
            pm_b_obs,
            vl_obs,
            vb_obs,
            proj_vl,
            proj_vb,
            delta_vl,
            delta_vb,
            delta_vT_gal,
        ]
    ).T


angmomdir = Path("/Volumes/newpassport/spheres")
made_spheres = [f.stem for f in angmomdir.glob("*.parquet")]

r_clu = pd.read_csv("/Users/sharifi/Documents/escapees/data/clu_params.csv")[
    ["name", "x", "y", "z", "v_rho", "v_phi", "v_z", "rho"]
]

r_clu = r_clu[~r_clu.name.isin(made_spheres)].reset_index(drop=True)

sphere_rad = 195

vTcols = [
    "vra",
    "vdec",
    "proj_vra",
    "proj_vdec",
    "proj_vrad",
    "delta_vra",
    "delta_vdec",
    "delta_vT",
    "pm_l_cosb_obs",
    "pm_b_obs",
    "vl_obs",
    "vb_obs",
    "proj_vl",
    "proj_vb",
    "delta_vl",
    "delta_vb",
    "delta_vT_gal",
]

con = duckdb.connect()

con.execute("""
CREATE VIEW allsky AS
SELECT * EXCLUDE ('__index_level_0__', 'source_id_1')
FROM '/Volumes/newpassport/allskylitjoin/*.parquet'
""")

for name, clu_x, clu_y, clu_z, clu_v_rho, clu_v_phi, clu_v_z, clu_rho in tqdm(
    r_clu.itertuples(index=False), total=len(r_clu)
):

    print(f"Starting query for {name}")

    df = con.execute(f"""
        SELECT *,
               sqrt((X - {clu_x})^2 +
                    (Y - {clu_y})^2 +
                    (Z - {clu_z})^2) AS dist_from_cluster
        FROM allsky
        WHERE (X - {clu_x})^2 +
              (Y - {clu_y})^2 +
              (Z - {clu_z})^2 <= {sphere_rad**2}
    """).df()

    print("Query done!")

    df["sphere_name"] = name

    n = len(df)
    clu_v_rho = np.full(n, clu_v_rho)
    clu_v_phi = np.full(n, clu_v_phi)
    clu_v_z = np.full(n, clu_v_z)
    clu_rho = np.full(n, clu_rho)

    df[vTcols] = skycoord_cylvel_ang_mom_to_dvT(
        df.ra.values,
        df.dec.values,
        df.parallax.values,
        df.pmra.values,
        df.pmdec.values,
        clu_v_rho,
        clu_v_phi,
        clu_v_z,
        clu_rho,
    )

    pq_path = f"/Volumes/newpassport/spheres/{name}.parquet"

    df.to_parquet(pq_path, compression="snappy")

    del df
