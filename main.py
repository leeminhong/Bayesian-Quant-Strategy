import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==============================================================================
# [1] 데이터 로더 클래스 (Data Handling)
# 설명: 야후 파이낸스에서 데이터를 가져오고, 전략에 필요한 보조지표를 계산합니다.
# ==============================================================================
class DataLoader:
    def __init__(self, symbol="NQ=F", interval="60m", start_date=None):
        self.symbol = symbol        # 거래 대상 (예: NQ=F)
        self.interval = interval    # 봉 주기 (예: 60분봉)
        self.start_date = start_date # 시작 날짜

    def fetch(self):
        # 1. 데이터 다운로드
        print(f"📥 [{self.symbol}] 데이터 다운로드 중... ({self.start_date} ~ )")
        df = yf.download(self.symbol, interval=self.interval, start=self.start_date, progress=False, auto_adjust=False)

        if df.empty:
            raise ValueError("데이터가 없습니다. 심볼이나 날짜를 확인하세요.")

        # 2. 컬럼명 정리 (MultiIndex 문제 해결 및 소문자 변환)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[0] in ['Open','High','Low','Close','Volume'] else col[1] for col in df.columns]
        df.columns = [c.lower() for c in df.columns]

        # 3. 수정 주가(Adj Close) 처리
        if 'adj close' in df.columns:
            df = df.drop(columns=['close', 'adj close'], errors='ignore')
            df = df.rename(columns={'adj close': 'close'})

        return df.dropna()

    @staticmethod
    def add_indicators(df):

        df = df.copy()

        # ---------------------------------------------------------
        # 1. 베이지안 확률 (Bayesian Probability)
        # ---------------------------------------------------------
        # 전봉 대비 상승이면 1, 아니면 0
        df['up'] = (df['close'] > df['close'].shift(1)).astype(int)
        # 라플라스 스무딩 적용: (1 + 상승횟수) / (2 + 전체기간)
        df['postMean'] = (1 + df['up'].rolling(120).sum()) / (2 + 120)

        # ---------------------------------------------------------
        # 2. Z-Score (표준화 점수)
        # ---------------------------------------------------------
        # 현재 가격이 평균 대비 몇 표준편차 떨어져 있는지 측정
        df['ma_z'] = df['close'].rolling(120).mean()
        df['std_z'] = df['close'].rolling(120).std(ddof=1) # ddof=1: 표본표준편차
        df['z'] = (df['close'] - df['ma_z']) / df['std_z']

        # ---------------------------------------------------------
        # 3. ATR (Average True Range) - 변동성 지표
        # ---------------------------------------------------------
        # 고가-저가, 고가-전일종가, 저가-전일종가 중 최대값
        df['tr'] = np.maximum(df['high'] - df['low'],
                   np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
        df['ATR'] = df['tr'].rolling(14).mean() # 단순이동평균(SMA) 방식 적용

        # ---------------------------------------------------------
        # 4. RSI (상대강도지수)
        # ---------------------------------------------------------
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ---------------------------------------------------------
        # 5. 기타 지표 (이동평균선 & 등락률)
        # ---------------------------------------------------------
        df['MA200'] = df['close'].rolling(200).mean()   # 장기 추세선
        df['rocDay'] = df['close'].pct_change(24) * 100 # 24시간 전 대비 등락률

        return df.dropna()

# ==============================================================================
# [2] 백테스팅 엔진 클래스 (Backtest Engine)
# 설명: 과거 데이터를 순회하며 매수/매도 로직을 실행하고 자산을 관리합니다.
# ==============================================================================
class Backtester:
    def __init__(self, df, start_capital=100000, point_value=20):
        self.df = df
        self.equity = start_capital      # 현재 자산
        self.start_capital = start_capital # 시작 자산
        self.point_value = point_value   # 1포인트당 가치 (승수)
        self.trades = []                 # 거래 기록 저장소
        self.equity_curve = []           # 자산 곡선 저장소

    def run(self):
        # 포지션 상태 변수 초기화
        position = 0       # 0:무포지션, 1:매수보유
        entry_price = 0.0  # 진입 가격
        entry_name = ""    # 진입 전략명 (Sniper/Surfer)
        highest_price = 0.0 # 진입 후 최고가 (트레일링 스탑용)

        # 데이터프레임 값을 넘파이 배열로 변환 (속도 최적화)
        times = self.df.index
        opens = self.df['open'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        closes = self.df['close'].values
        atrs = self.df['ATR'].values
        postMeans = self.df['postMean'].values
        zs = self.df['z'].values
        rsis = self.df['RSI'].values
        ma200s = self.df['MA200'].values
        rocs = self.df['rocDay'].values

        print("⚙️ 백테스팅 엔진 가동...")

        # ---------------------------------------------------------
        # 봉 단위 시뮬레이션 루프 (Loop)
        # ---------------------------------------------------------
        for i in range(len(self.df) - 1):
            curr_time = times[i]
            close = closes[i]
            atr = atrs[i]

            # [중요] 미래 참조 방지: i번째 봉을 보고 판단하여 -> i+1번째 시가에 진입
            next_open = opens[i+1]

            # -----------------------------------------------------
            # [A] 진입 전략 조건 (Entry Logic)
            # -----------------------------------------------------

            # 1. Sniper (역추세): 급락 후 반등을 노림
            # 조건: 24시간등락률 > -3%, 확률 40% 이상, Z점수 <-2.0 (과매도), RSI < 30 (침체)
            # 추가조건: 종가가 저가 대비 30% 이상 반등 (아랫꼬리)
            cond_sniper = (rocs[i] > -3.0) and (postMeans[i] >= 0.4) and (zs[i] < -2.0) and (rsis[i] < 30) and (close > lows[i] + (highs[i]-lows[i])*0.3)

            # 2. Surfer (추세 추종): 상승 추세 눌림목을 노림
            # 조건: 24시간등락률 > -3%, 200일선 위, 확률 55% 이상, Z점수 < 0 (눌림목), RSI < 60 (과열아님)
            cond_surfer = (rocs[i] > -3.0) and (close > ma200s[i]) and (postMeans[i] >= 0.55) and (zs[i] < 0) and (rsis[i] < 60)

            # 우선순위: Sniper가 발생하면 Surfer는 무시 (중복 진입 방지)
            if cond_sniper: cond_surfer = False

            # -----------------------------------------------------
            # [B] 진입 실행 (Execution)
            # -----------------------------------------------------
            if position == 0:
                if cond_sniper:
                    position = 1
                    entry_price = next_open
                    entry_name = "B_Sniper"
                    highest_price = next_open # 최고가 초기화
                elif cond_surfer:
                    position = 1
                    entry_price = next_open
                    entry_name = "B_Surfer"
                    highest_price = next_open

            # -----------------------------------------------------
            # [C] 청산 관리 (Exit & Risk Management)
            # -----------------------------------------------------
            elif position == 1:
                # 최고가 갱신 (트레일링 스탑 기준점)
                if close > highest_price: highest_price = close

                # [트레일링 스탑 설정]
                # Sniper: 변동성 3배 여유 (역추세라 흔들림 감수)
                # Surfer: 변동성 1배 타이트하게
                mult = 3.0 if entry_name == "B_Sniper" else 1.0
                trail_stop = highest_price - (mult * atr)

                exit_p = 0.0
                reason = ""

                # 1. 트레일링 스탑 청산 (다음 봉 저가가 스탑 라인 건드림)
                if lows[i+1] <= trail_stop:
                    exit_p = min(next_open, trail_stop) # 갭 하락 시 시가 청산
                    reason = "TrailStop"

                # 2. 베이지안 확률 약세 청산 (확률 30% 미만 시 탈출)
                elif postMeans[i] < 0.35:
                    exit_p = next_open
                    reason = "BayesLow"

                # 청산 확정 및 손익 계산
                if exit_p > 0:
                    pnl = (exit_p - entry_price) * self.point_value # 승수 적용
                    self.equity += pnl

                    self.trades.append({
                        'Date': times[i+1],
                        'Type': entry_name,
                        'Entry': entry_price,
                        'Exit': exit_p,
                        'PnL': pnl,
                        'Reason': reason
                    })
                    position = 0      # 포지션 초기화
                    highest_price = 0 # 최고가 초기화

            # 매 봉 마감 시 자산 기록
            self.equity_curve.append({'Date': times[i], 'Equity': self.equity})

    def get_results(self):
        return pd.DataFrame(self.trades), pd.DataFrame(self.equity_curve).set_index('Date')

# ==============================================================================
# [3] 성과 분석기 클래스 (Performance Analyzer)
# 설명: 백테스트 결과를 받아 MDD, Sharpe Ratio 등 전문 지표를 산출합니다.
# ==============================================================================
class PerformanceAnalyzer:
    @staticmethod
    def analyze(trades_df, equity_df, start_capital):
        if trades_df.empty: return "거래 내역 없음"

        # 1. 기본 통계
        total_pnl = trades_df['PnL'].sum()
        win_rate = (trades_df['PnL'] > 0).mean() * 100
        avg_pnl = trades_df['PnL'].mean()

        # Profit Factor (총이익 / 총손실)
        gross_profit = trades_df[trades_df['PnL']>0]['PnL'].sum()
        gross_loss = abs(trades_df[trades_df['PnL']<0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 2. MDD (최대 낙폭) 계산
        equity_df['Peak'] = equity_df['Equity'].cummax() # 전고점
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
        mdd = equity_df['Drawdown'].min()

        # 3. Sharpe Ratio (위험 대비 수익률)
        daily_ret = equity_df['Equity'].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0

        # 결과 출력
        print("\n" + "="*40)
        print(" 📊 PERFORMANCE REPORT (성과 보고서)")
        print("="*40)
        print(f"Initial Capital : ${start_capital:,.0f}")
        print(f"Final Equity    : ${equity_df['Equity'].iloc[-1]:,.0f}")
        print(f"Net Profit      : ${total_pnl:,.2f} ({total_pnl/start_capital*100:.2f}%)")
        print(f"Total Trades    : {len(trades_df)}")
        print(f"Win Rate        : {win_rate:.2f}%")
        print(f"Profit Factor   : {profit_factor:.2f}")
        print(f"MDD             : {mdd:.2f}%")
        print(f"Sharpe Ratio    : {sharpe:.2f}")
        print("="*40)

        return equity_df

    @staticmethod
    def plot_equity(equity_df):
        plt.figure(figsize=(12, 8))

        # 자산 곡선 (Equity Curve)
        plt.subplot(2, 1, 1)
        plt.plot(equity_df.index, equity_df['Equity'], label='Equity', color='blue')
        plt.title('Equity Curve')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # 낙폭 곡선 (Drawdown)
        plt.subplot(2, 1, 2)
        plt.fill_between(equity_df.index, equity_df['Drawdown'], 0, color='red', alpha=0.3)
        plt.plot(equity_df.index, equity_df['Drawdown'], color='red', linewidth=0.8)
        plt.title('Drawdown (%)')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

# ==============================================================================
# [4] 메인 실행부 (Main Execution)
# ==============================================================================
if __name__ == "__main__":
    # 사용자 설정 변수
    SYMBOL = "NQ=F"       # 심볼: 나스닥 선물
    START = "2025-03-01"  # 시작일
    CAPITAL = 100000      # 시작 자본금 ($)
    POINT_VAL = 20        # 포인트당 가치 (NQ=20, MNQ=2)

    # 1. 데이터 로드 및 지표 생성
    loader = DataLoader(SYMBOL, "60m", START)
    try:
        raw_df = loader.fetch()
        data = DataLoader.add_indicators(raw_df)

        # 2. 백테스트 실행 (엔진 가동)
        engine = Backtester(data, CAPITAL, POINT_VAL)
        engine.run()
        trades, equity = engine.get_results()

        # 3. 성과 분석 및 결과 출력
        if not trades.empty:
            equity_with_dd = PerformanceAnalyzer.analyze(trades, equity, CAPITAL)

            # 4. 그래프 그리기
            PerformanceAnalyzer.plot_equity(equity_with_dd)

            # 5. CSV 파일로 결과 저장
            trades.to_csv("Portfolio_Backtest_Result.csv")
            print("\n✅ 결과 저장 완료: Portfolio_Backtest_Result.csv")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
