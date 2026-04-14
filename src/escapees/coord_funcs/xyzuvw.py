def xyzuvw(ra, dec, parallax, pmra, pmdec, radial_velocity):
    """
        Converts ICRS ra (deg), dec (deg), parallax (mas), pmra (mas/yr),
        pmdec (mas/yr), and radial velocity (km/s) to heliocentric
        Cartesian XYZ (pc) and UVW (km/s)

    Returns:
        numpy.ndarray : (n x 6) XYZUVW array
    """

    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Galactic

    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=1000 / parallax * u.pc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=radial_velocity * u.km / u.s,
        frame="icrs",
    ).transform_to(Galactic())

    return np.hstack(
        [
            coords.cartesian.xyz.to(u.pc).value.T,
            coords.cartesian.differentials["s"].d_xyz.to(u.km / u.s).value.T,
        ]
    )
