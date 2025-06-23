from config import (                         
    crypto_tk, stock_tk, index_tk,
    START_H, END_H, H,
    DATA_LAKE_DIR,
)
from features import Par_, raw, index_df, FE 
  
from stratagies import (                      
    generate_all_nf_signals,
    generate_all_cb_signals,
    generate_all_tsai_signals,
    generate_all_cql_signals,
)

# import logging
# log = logging.getLogger(__name__)

def main():

    # par = Par_(crypto_tk, stock_tk, index_tk, START_H, END_H)
    # raw_df = par.load()
    # fe = FE(
    #     tickers=crypto_tk + stock_tk,
    #     index_df=index_df,
    #     lake=DATA_LAKE_DIR,
    #     base_win=15,
    #     thr=3.0,
    #     vol_window=35,
    #     min_win=5,
    #     max_win=35
    # )
    # fe.run(raw_df)   

    signals = {
        **dict(zip(
            ["ptst_val","ptst_test","nhits_val","nhits_test"],
            generate_all_nf_signals(fe, stock_tk, H)
        )),
        **dict(zip(
            ["cb_val","cb_test"],
            generate_all_cb_signals(fe, stock_tk, H)
        )),
        **dict(zip(
            ["tsi_val","tsi_test","tstp_val","tstp_test","mr_val","mr_test","tsi_tr","tstp_tr"],
            generate_all_tsai_signals(fe, stock_tk, H)
        )),
        **dict(zip(
            ["cql_val","cql_test"],
            generate_all_cql_signals(fe, stock_tk, H)
        ))
    }
    
    return signals

if __name__ == "__main__":
    main()
