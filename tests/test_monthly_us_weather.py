import unittest
from datetime import date

from weather.backfill_monthly_us_weather import (
    REGION_MEMBERS,
    STATE_NOAA_CODES,
    parse_dc_station,
    parse_population,
    parse_state_file,
)


class MonthlyWeatherParserTests(unittest.TestCase):
    def test_noaa_state_record_parsing_and_sentinel(self):
        # state=AL, division=0, element=02 (Tavg), year=1973
        months = ["  50.0", "  51.0", "-99.90"] + ["  55.0"] * 9
        line = "0010021973" + "".join(value.rjust(7) for value in months)
        parsed = parse_state_file((line + "\n").encode(), "tmpcst")
        self.assertEqual(parsed[("AL", date(1973, 1, 1))], 50.0)
        self.assertNotIn(("AL", date(1973, 3, 1)), parsed)

    def test_population_extracts_2010_decennial_value(self):
        lines = []
        # Two Alabama divisions sum to the expected test weight for 2010.
        lines.append("AL 0101 " + " ".join(str(x) for x in range(11)) + " 400 500")
        lines.append("AL 0102 " + " ".join(str(x) for x in range(11)) + " 600 700")
        # Supply one division for every other state with fields through 2020.
        for abbrev in set(STATE_NOAA_CODES) - {"AL"}:
            lines.append(f"{abbrev} 0101 " + " ".join(str(x) for x in range(13)))
        populations = parse_population(("\n".join(lines) + "\n").encode())
        self.assertEqual(populations["AL"], 1000)
        self.assertEqual(len(populations), 48)

    def test_census_region_crosswalk_is_contiguous_48_plus_dc(self):
        members = [state for states in REGION_MEMBERS.values() for state in states]
        self.assertEqual(len(members), 49)
        self.assertEqual(len(set(members)), 49)
        self.assertEqual(members.count("DC"), 1)
        self.assertNotIn("AK", members)
        self.assertNotIn("HI", members)

    def test_dc_daily_extremes_become_monthly_degree_days(self):
        # Two valid days, 10.0C max and 0.0C min => daily mean 41F.
        def ghcn_row(element, values):
            chunks = []
            for value in values:
                chunks.append(f"{value:5d}   ")
            return "USW00013743" + "202001" + element + "".join(chunks)

        tmax = [100, 100] + [-9999] * 29
        tmin = [0, 0] + [-9999] * 29
        raw = (ghcn_row("TMAX", tmax) + "\n" + ghcn_row("TMIN", tmin) + "\n").encode()
        result = parse_dc_station(raw)
        tavg, hdd, cdd = result[date(2020, 1, 1)]
        self.assertAlmostEqual(tavg, 41.0)
        self.assertAlmostEqual(hdd, 48.0)
        self.assertAlmostEqual(cdd, 0.0)


if __name__ == "__main__":
    unittest.main()
