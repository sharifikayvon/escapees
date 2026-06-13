def radecplxuvw_to_vT(ra, dec, parallax, U, V, W):
    """Takes ICRS ra (deg), dec (deg), parallax (mas),
    and UVW (km/s) vectors, and outputs proper motions and
    transverse velocities.

    Args:
        ra (float): _description_
        dec (float): _description_
        parallax (float): _description_
        U (float): _description_
        V (float): _description_
        W (float): _description_

    Returns:
        numpy.ndarray: shape (n x 4) pmra, pmdec, vra, vdec

    """
    gal = SkyCoord(
        ra=ra * u.deg, dec=dec * u.deg, distance=(1000 / parallax) * u.pc, frame=ICRS()
    ).transform_to(Galactic())

    gal = gal.realize_frame(
        gal.data.with_differentials(
            CartesianDifferential(
                d_x=U * u.km / u.s, d_y=V * u.km / u.s, d_z=W * u.km / u.s
            )
        )
    )

    icrs = gal.transform_to(ICRS())
    pmra = icrs.pm_ra_cosdec.value
    pmdec = icrs.pm_dec.value
    vra = 4.74047 * pmra / parallax
    vdec = 4.74047 * pmdec / parallax

    out = np.vstack([pmra, pmdec, vra, vdec]).T

    return out
