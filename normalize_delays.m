function normalized = normalize_delays(delays)
    Frequancy = 4.464e6; 
    delays = delays - min(min(delays));
    delays = unwrap(delays);
    delays = delays / (2*pi);
    normalized = delays / Frequancy;

end