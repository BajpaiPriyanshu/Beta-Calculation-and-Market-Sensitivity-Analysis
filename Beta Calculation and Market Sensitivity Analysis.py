import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== Beta Calculation and Market Sensitivity Analysis ===\n")

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=252, freq='B')

nifty_returns = np.random.normal(0.0008, 0.015, 252)
sensex_returns = np.random.normal(0.0007, 0.014, 252)

stocks = {
    'RELIANCE': np.random.normal(0.0006, 0.018, 252) + 1.2 * nifty_returns + np.random.normal(0, 0.008, 252),
    'TCS': np.random.normal(0.0008, 0.012, 252) + 0.8 * nifty_returns + np.random.normal(0, 0.006, 252),
    'HDFC_BANK': np.random.normal(0.0007, 0.016, 252) + 1.1 * nifty_returns + np.random.normal(0, 0.007, 252),
    'INFOSYS': np.random.normal(0.0009, 0.014, 252) + 0.9 * nifty_returns + np.random.normal(0, 0.006, 252),
    'ITC': np.random.normal(0.0005, 0.013, 252) + 0.6 * nifty_returns + np.random.normal(0, 0.005, 252)
}

returns_data = pd.DataFrame({
    'Date': dates,
    'NIFTY': nifty_returns,
    'SENSEX': sensex_returns,
    **stocks
})
returns_data.set_index('Date', inplace=True)

print("Sample of Returns Data:")
print(returns_data.head())
print(f"\nDataset shape: {returns_data.shape}")

print("\n=== BETA CALCULATION AGAINST NIFTY ===")

def calculate_beta(stock_returns, market_returns):
    clean_data = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data['market'], clean_data['stock'])
    return {
        'beta': slope,
        'alpha': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_error': std_err
    }

beta_results_nifty = {}
stock_names = ['RELIANCE', 'TCS', 'HDFC_BANK', 'INFOSYS', 'ITC']

for stock in stock_names:
    beta_info = calculate_beta(returns_data[stock], returns_data['NIFTY'])
    beta_results_nifty[stock] = beta_info
    print(f"\n{stock}:")
    print(f"  Beta: {beta_info['beta']:.3f}")
    print(f"  Alpha: {beta_info['alpha']:.5f}")
    print(f"  R-squared: {beta_info['r_squared']:.3f}")
    print(f"  P-value: {beta_info['p_value']:.5f}")

print("\n=== BETA RANKINGS AND MARKET SENSITIVITY ANALYSIS ===")

beta_df = pd.DataFrame({
    'Stock': stock_names,
    'Beta': [beta_results_nifty[stock]['beta'] for stock in stock_names],
    'Alpha': [beta_results_nifty[stock]['alpha'] for stock in stock_names],
    'R_Squared': [beta_results_nifty[stock]['r_squared'] for stock in stock_names],
    'Volatility': [returns_data[stock].std() for stock in stock_names]
})

beta_df_sorted = beta_df.sort_values('Beta', ascending=False).reset_index(drop=True)
beta_df_sorted['Rank'] = range(1, len(beta_df_sorted) + 1)

print("Beta Rankings (Against NIFTY):")
print("="*60)
for _, row in beta_df_sorted.iterrows():
    risk_level = "HIGH RISK" if row['Beta'] > 1.0 else "LOW RISK"
    print(f"Rank {row['Rank']}: {row['Stock']:<12} | Beta: {row['Beta']:.3f} | {risk_level}")

print(f"\n=== MARKET SENSITIVITY CLASSIFICATION ===")

def classify_beta(beta_value):
    if beta_value > 1.2:
        return "Highly Aggressive"
    elif beta_value > 1.0:
        return "Moderately Aggressive"
    elif beta_value > 0.8:
        return "Moderately Defensive"
    else:
        return "Highly Defensive"

beta_df_sorted['Classification'] = beta_df_sorted['Beta'].apply(classify_beta)

print("Stock Classifications:")
for _, row in beta_df_sorted.iterrows():
    print(f"{row['Stock']:<12}: {row['Classification']:<20} (Beta: {row['Beta']:.3f})")

print(f"\n=== STATISTICAL SUMMARY ===")
print(f"Average Beta: {beta_df['Beta'].mean():.3f}")
print(f"Beta Standard Deviation: {beta_df['Beta'].std():.3f}")
print(f"Highest Beta: {beta_df['Beta'].max():.3f} ({beta_df.loc[beta_df['Beta'].idxmax(), 'Stock']})")
print(f"Lowest Beta: {beta_df['Beta'].min():.3f} ({beta_df.loc[beta_df['Beta'].idxmin(), 'Stock']})")

equal_weight = 1/len(stock_names)
portfolio_beta = sum(beta_df['Beta'] * equal_weight)
print(f"Equal-Weight Portfolio Beta: {portfolio_beta:.3f}")

print(f"\n=== CREATING VISUALIZATIONS ===")

import seaborn as sns

plt.figure(figsize=(10, 6))
colors = ['red' if beta > 1.0 else 'green' for beta in beta_df_sorted['Beta']]
plt.bar(beta_df_sorted['Stock'], beta_df_sorted['Beta'], color=colors, alpha=0.7)
plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='Market Beta = 1.0')
plt.title('Beta Coefficients by Stock', fontweight='bold')
plt.ylabel('Beta Value')
plt.xlabel('Stocks')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(beta_df['Beta'], beta_df['Volatility'], s=100, alpha=0.7, c='blue')
for i, stock in enumerate(beta_df['Stock']):
    plt.annotate(stock, (beta_df['Beta'].iloc[i], beta_df['Volatility'].iloc[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.title('Risk vs Beta Relationship', fontweight='bold')
plt.xlabel('Beta (Market Sensitivity)')
plt.ylabel('Volatility (Standard Deviation)')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
sample_stock = 'RELIANCE'
plt.scatter(returns_data['NIFTY'], returns_data[sample_stock], alpha=0.6, s=20)
z = np.polyfit(returns_data['NIFTY'], returns_data[sample_stock], 1)
p = np.poly1d(z)
plt.plot(returns_data['NIFTY'], p(returns_data['NIFTY']), "r--", alpha=0.8,
         label=f'Beta = {beta_results_nifty[sample_stock]["beta"]:.3f}')
plt.title(f'{sample_stock} vs NIFTY Returns', fontweight='bold')
plt.xlabel('NIFTY Daily Returns')
plt.ylabel(f'{sample_stock} Daily Returns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(beta_df['Beta'], kde=True, bins=10, color='purple', edgecolor='black')
plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='Market Beta = 1.0')
plt.axvline(x=beta_df['Beta'].mean(), color='orange', linestyle='-', alpha=0.8,
            label=f'Average = {beta_df["Beta"].mean():.3f}')
plt.title('Distribution of Beta Values', fontweight='bold')
plt.xlabel('Beta Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n=== INVESTMENT RECOMMENDATIONS ===")
print("Based on Beta Analysis:")
print("-" * 50)

high_beta_stocks = beta_df_sorted[beta_df_sorted['Beta'] > 1.0]['Stock'].tolist()
low_beta_stocks = beta_df_sorted[beta_df_sorted['Beta'] < 1.0]['Stock'].tolist()

print("HIGH BETA STOCKS (Beta > 1.0) - Aggressive Growth:")
for stock in high_beta_stocks:
    beta_val = beta_df_sorted[beta_df_sorted['Stock'] == stock]['Beta'].iloc[0]
    print(f"  • {stock}: Beta {beta_val:.3f} - Higher returns in bull markets, higher losses in bear markets")

print(f"\n LOW BETA STOCKS (Beta < 1.0) - Defensive/Stable:")
for stock in low_beta_stocks:
    beta_val = beta_df_sorted[beta_df_sorted['Stock'] == stock]['Beta'].iloc[0]
    print(f"  • {stock}: Beta {beta_val:.3f} - Lower volatility, more stable during market downturns")

print(f"\n PORTFOLIO STRATEGY RECOMMENDATIONS:")
print(f"  • Bull Market: Favor high-beta stocks ({', '.join(high_beta_stocks)})")
print(f"  • Bear Market: Favor low-beta stocks ({', '.join(low_beta_stocks)})")
print(f"  • Balanced Portfolio: Mix of both for optimal risk-return profile")

print(f"\n=== ANALYSIS COMPLETE ===")
print("Key Takeaways:")
print("1. Beta measures a stock's sensitivity to market movements")
print("2. Beta > 1.0 means more volatile than market")
print("3. Beta < 1.0 means less volatile than market")
print("4. Use beta for portfolio construction and risk management")
