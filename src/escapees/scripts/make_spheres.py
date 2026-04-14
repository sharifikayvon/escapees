import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm
from escapees.coord_funcs import uvw_to_vT, radecplxuvw_to_vT, xyzuvw

r_clu_cols = ["name", "x", "y", "z", "u", "v", "w"]

r_clu = pd.read_csv("../data/clu_params.csv")[r_clu_cols]

sphere_rad = 180

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


for name, clu_x, clu_y, clu_z, clu_u, clu_v, clu_w in tqdm(
    r_clu.itertuples(index=False), total=288
):

    clu_xyz = np.array([clu_x, clu_y, clu_z])

    query = f"""
    SELECT *
    FROM '/Volumes/travelpassport/tables/allskylitjoin/*.parquet'
    WHERE sqrt((X - {clu_x})**2 + (Y - {clu_y})**2 + (Z - {clu_z})**2) <= {sphere_rad}
    """

    print(f"Starting query for {name}")
    df = con.execute(query).df()
    print("Query done!")
    df["sphere_name"] = name

    sphere_xyz = df[["X", "Y", "Z"]].to_numpy()
    df["dist_from_cluster"] = np.linalg.norm(sphere_xyz - clu_xyz, axis=1)

    clu_uvw = np.tile([clu_u, clu_v, clu_w], (len(df), 1))

    df[vTcols] = uvw_to_vT(
        df.ra.to_numpy(),
        df.dec.to_numpy(),
        df.parallax.to_numpy(),
        df.pmra.to_numpy(),
        df.pmdec.to_numpy(),
        clu_uvw[:, 0],
        clu_uvw[:, 1],
        clu_uvw[:, 2],
    )

    print(f"Saving to {name}.parquet")

    pq_path = f"/Volumes/travelpassport/spheres/{name}.parquet"

    df.to_parquet(pq_path, compression="snappy")
    print("Done!")

    del df
