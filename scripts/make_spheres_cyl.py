import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm

r_clu = pd.read_csv("/Users/sharifi/Documents/escapees/data/clu_params.csv")[
    ["name", "x", "y", "z", "v_rho", "v_phi", "v_z"]
]

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
]

con = duckdb.connect()

con.execute("""
CREATE VIEW allsky AS
SELECT * EXCLUDE ('__index_level_0__', 'source_id_1')
FROM '/Volumes/travelpassport/tables/allskylitjoin/*.parquet'
""")

for name, clu_x, clu_y, clu_z, clu_v_rho, clu_v_phi, clu_v_z in tqdm(
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

    df[vTcols] = skycoord_cylvel_to_dvT(
        df.ra.values,
        df.dec.values,
        df.parallax.values,
        df.pmra.values,
        df.pmdec.values,
        clu_v_rho,
        clu_v_phi,
        clu_v_z,
    )

    pq_path = f"/Volumes/travelpassport/tables/spheres_cylvel/{name}.parquet"

    df.to_parquet(pq_path, compression="snappy")

    del df


def skycoord_cylvel_to_dvT(ra, dec, parallax, pmra, pmdec, v_rho, v_phi, v_z):

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
    v_z = np.asarray(v_z) * u.km / u.s

    vx = v_rho * np.cos(phi) - v_phi * np.sin(phi)
    vy = v_rho * np.sin(phi) + v_phi * np.cos(phi)
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

    vra = 4.74047 * pmra / parallax
    vdec = 4.74047 * pmdec / parallax

    proj_pmra = icrs.pm_ra_cosdec.to(u.mas / u.yr).value
    proj_pmdec = icrs.pm_dec.to(u.mas / u.yr).value
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

    return np.vstack(
        [vra, vdec, proj_vra, proj_vdec, proj_vrad, delta_vra, delta_vdec, delta_vT]
    ).T
