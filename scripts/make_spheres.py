import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm
from escapees.coord_funcs import uvw_to_vT

r_clu = pd.read_csv("../data/clu_params.csv")[["name", "x", "y", "z", "u", "v", "w"]]

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

con.execute(
    """
CREATE VIEW allsky AS
SELECT * FROM '/Volumes/travelpassport/tables/allskylitjoin/*.parquet'
"""
)

for name, clu_x, clu_y, clu_z, clu_u, clu_v, clu_w in tqdm(
    r_clu.itertuples(index=False), total=len(r_clu)
):

    print(f"Starting query for {name}")

    df = con.execute(
        f"""
        SELECT *,
               sqrt((X - {clu_x})^2 +
                    (Y - {clu_y})^2 +
                    (Z - {clu_z})^2) AS dist_from_cluster
        FROM allsky
        WHERE (X - {clu_x})^2 +
              (Y - {clu_y})^2 +
              (Z - {clu_z})^2 <= {sphere_rad**2}
    """
    ).df()

    print("Query done!")

    if len(df) == 0:
        continue

    df["sphere_name"] = name

    n = len(df)
    U = np.full(n, clu_u)
    V = np.full(n, clu_v)
    W = np.full(n, clu_w)

    df[vTcols] = uvw_to_vT(
        df.ra.values,
        df.dec.values,
        df.parallax.values,
        df.pmra.values,
        df.pmdec.values,
        U,
        V,
        W,
    )

    pq_path = f"/Volumes/travelpassport/spheres/{name}.parquet"

    df.to_parquet(pq_path, compression="snappy")

    del df
