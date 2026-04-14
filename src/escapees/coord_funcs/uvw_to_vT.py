def uvw_to_vT(ra, dec, parallax, pmra, pmdec, U, V, W):
    """Takes ICRS ra (deg), dec (deg), parallax (mas), pmra (mas/yr),
        pmdec (mas/yr), and UVW (km/s) vectors, and compares how similar
        a star's observed astrometry is to the UVW vector at its position.

    Args:
        ra (float): _description_
        dec (float): _description_
        parallax (float): _description_
        pmra (float): _description_
        pmdec (float): _description_
        U (float): _description_
        V (float): _description_
        W (float): _description_

    Returns:
        numpy.ndarray: shape (n x 7) vra, vdec, proj_vra, proj_vdec, delta_vra, delta_vdec, delta_vT

    """

    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord, CartesianDifferential, Galactic, ICRS

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

    proj_vra = 4.74047 * icrs.pm_ra_cosdec.value / parallax
    proj_vdec = 4.74047 * icrs.pm_dec.value / parallax
    proj_vrad = icrs.radial_velocity.to(u.km / u.s).value

    vra = 4.74047 * pmra / parallax
    vdec = 4.74047 * pmdec / parallax

    delta_vra = vra - proj_vra
    delta_vdec = vdec - proj_vdec
    delta_vT = np.sqrt(delta_vra**2 + delta_vdec**2)

    out = np.vstack(
        [vra, vdec, proj_vra, proj_vdec, proj_vrad, delta_vra, delta_vdec, delta_vT]
    ).T

    return out


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
