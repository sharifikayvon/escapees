def galcen_cyl(ra, dec, parallax, pmra, pmdec, radial_velocity):
    """
    Converts ICRS coordinates to Galactocentric cylindrical coordinates.

    Returns
    -------
    numpy.ndarray
        (n x 6) array containing

        rho      : kpc
        phi      : rad
        z        : kpc
        v_rho    : km/s
        v_phi    : km/s
        v_z      : km/s
    """

    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Galactocentric

    c = (
        SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            distance=1000 / parallax * u.pc,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            radial_velocity=radial_velocity * u.km / u.s,
            frame="icrs",
        )
        .transform_to(Galactocentric())
        .cylindrical
    )

    return np.column_stack(
        [
            c.rho.to_value(u.kpc),
            c.phi.to_value(u.rad),
            c.z.to_value(u.kpc),
            c.velocity.d_rho.to_value(u.km / u.s),
            c.velocity.d_phi.to_value(u.rad / u.Myr),
            c.velocity.d_z.to_value(u.km / u.s),
        ]
    )


# To get v_phi in km/s, (d_phi*u.rad/u.Myr*rho*u.kpc).to(u.km/u.s)
